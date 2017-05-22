# -*- coding: utf-8 -*-
# 拥有GUI时，可以运行display.py，通过pygame实时更新显示test的结果；
# 而在服务器远程运行时，pygame无法使用则改用opencv绘制图像，然后直接写入视频文件以备查看。

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
import cv2
import os, sys
from collections import deque

from environment.environment import Environment
from model.model import UnrealModel
from constants import *
from train.experience import ExperienceFrame

from maze_map import MazeMap

SAVE_DIR = './'
HEIGHT = 400
WIDTH = 400
FPS = 15

BLUE  = (128, 128, 255)
RED   = (255, 192, 192)
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)

class MovieWriter(object):
  def __init__(self, file_name, frame_size, fps):
    """
    frame_size is (w, h)
    """
    self._frame_size = frame_size
    fourcc = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')
    self.vout = cv2.VideoWriter()
    success = self.vout.open(file_name, fourcc, fps, frame_size, True)
    if not success:
      print("Create movie failed: {0}".format(file_name))

  def add_frame(self, frame):
    """
    frame shape is (h, w, 3), dtype is np.uint8
    """
    self.vout.write(frame)

  def close(self):
    self.vout.release()
    self.vout = None


class StateHistory(object):
  def __init__(self):
    self._states = deque(maxlen=3)

  def add_state(self, state):
    self._states.append(state)

  @property
  def is_full(self):
    return len(self._states) >= 3

  @property
  def states(self):
    return list(self._states)

class ValueHistory(object):
  def __init__(self):
    self._values = deque(maxlen=100)

  def add_value(self, value):
    self._values.append(value)

  @property
  def is_empty(self):
    return len(self._values) == 0

  @property
  def values(self):
    return self._values


class Tester(object):
  def __init__(self):
    self.img = np.zeros(shape=(HEIGHT, WIDTH, 3), dtype=np.uint8)
    self.action_size = Environment.get_action_size()
    self.global_network = UnrealModel(self.action_size, -1, "/cpu:0", for_display=True)
    self.env = Environment.create_environment()
    self.value_history = ValueHistory()
    self.state_history = StateHistory()
    self.ep_reward = 0
    self.mazemap = MazeMap()

  def process(self, sess):
    self.img = np.zeros(shape=(HEIGHT, WIDTH, 3), dtype=np.uint8)
    last_action = self.env.last_action
    last_reward = np.clip(self.env.last_reward, -1, 1)
    last_action_reward = ExperienceFrame.concat_action_and_reward(last_action, 
                                                                  self.action_size,
                                                                  last_reward)
    if not USE_PIXEL_CHANGE:
      pi_values, v_value = self.global_network.run_base_policy_and_value(sess,
                                                                        self.env.last_state,
                                                                        last_action_reward)
    else:
      pi_values, v_value, pc_q = self.global_network.run_base_policy_value_pc_q(sess,
                                                                                self.env.last_state,
                                                                                last_action_reward)
    self.value_history.add_value(v_value)
    action = self.choose_action(pi_values)
    state, reward, terminal, pc, vtrans, vrot = self.env.process(action)
    self.state_history.add_state(state)
    self.ep_reward += reward
    self.mazemap.update(vtrans, vrot)
    if reward > 9: # agent到达迷宫终点时，reward为10，地图需要重置
      self.mazemap.reset()
    if terminal: # lab环境默认3600帧为一个episode而不是到达迷宫终点时给terminal信号
      self.env.reset()
      self.ep_reward = 0
      self.mazemap.reset()

    self.show_ob(state, 3, 3, "Observation")
    self.show_pc(pc, 100, 3, 3.0, "Pixel Change")
    self.show_pc(pc_q[:,:,action], 200, 3, 0.4, "PC Q")
    self.show_map(300, 3, "Maze Map")
    self.show_pi(pi_values)
    self.show_reward()
    self.show_rp()
    self.show_value()

  def choose_action(self, pi_values):
    return np.random.choice(range(len(pi_values)), p=pi_values)
  
  def scale_image(self, image, scale):
    return image.repeat(scale, axis=0).repeat(scale, axis=1)

  def draw_text(self, text, left, bottom, color=WHITE):
    font = cv2.FONT_HERSHEY_COMPLEX
    cv2.putText(self.img, text, (left, bottom), font, 0.35, color)

  def show_pc(self, pc, left, top, rate, label):
    pc = np.clip(pc * 255.0 * rate, 0.0, 255.0)
    data = pc.astype(np.uint8)
    data = np.stack([data for _ in range(3)], axis=2)
    data = self.scale_image(data, 4)
    h = data.shape[0]
    w = data.shape[1]
    self.img[top:top+h, left:left+w, :] = data
    self.draw_text(label, (left+2), (top+h+15))

  def show_map(self, left, top, label):
    maze = self.mazemap.get_map(84, 84)
    maze = (maze * 255).astype(np.uint8)
    h = maze.shape[0]
    w = maze.shape[1]
    self.img[top:top+h, left:left+w, :] = maze
    self.draw_text(label, (left+2), (top+h+5))
  
  def show_pi(self, pi):
    for i in range(len(pi)):
      width = int(pi[i]*100)
      cv2.rectangle(self.img, (3, 113+15*i), (width, 120+15*i), WHITE)
    self.draw_text("Policy", 20, 120+15*len(pi))

  def show_ob(self, state, left, top, label):
    state = (state*255.0).astype(np.uint8)
    h = state.shape[0]
    w = state.shape[1]
    self.img[top:top+h, left:left+w, :] = state
    self.draw_text(label, (left+2), (top+h+15))

  def show_value(self, left, top, height, width):
    if self.value_history.is_empty:
      return

    min_v = float("inf")
    max_v = float("-inf")
    values = self.value_history.values

    for v in values:
      min_v = min(min_v, v)
      max_v = max(max_v, v)

    bottom = top + height
    right = left + width
    d = max_v - min_v
    last_r = 0.0
    for i,v in enumerate(values):
      r = (v-min_v) / d
      if i > 0:
        x0 = i-1 + left
        x1 = i   + left
        y0 = bottom - last_r * height
        y1 = bottom - r * height
        cv2.line(self.img, (y0, x0), (y1, x1), BLUE, 2)
      last_r = r

    cv2.line(self.img, (top, left), (bottom, left), WHITE, 1)
    cv2.line(self.img, (top, right), (bottom, right), WHITE, 1)
    cv2.line(self.img, (top, left), (top, right), WHITE, 1)
    cv2.line(self.img, (bottom, left), (bottom, right), WHITE, 1)
    self.draw_text("Q Value", 120, 215)


  def show_rp(self):
    pass

  def show_reward(self):
    self.draw_text("Reward: {}".format(int(self.ep_reward)), 10, 230)

  def get_frame(self):
    return self.img

if __name__ == '__main__':
  sess = tf.Session()
  init = tf.global_variables_initializer()
  sess.run(init)

  tester = Tester()
  saver = tf.train.Saver()
  ckpt = tf.train.get_checkpoint_state(CHECKPOINT_DIR)
  if ckpt and ckpt.model_checkpoint_path:
    saver.restore(sess, ckpt.model_checkpoint_path)
    print("Checkpoint loaded:", ckpt.model_checkpoint_path)
  else:
    print("Could not find checkpoint")

  writer = MovieWriter(SAVE_DIR+"out.avi", (HEIGHT, WIDTH), FPS)
  print("start recording video...")
  for i in range(FPS*30):
    tester.process(sess)
    frame = tester.get_frame()
    writer.add_frame(frame)
    if i % FPS == 0:
      print(str(i/FPS)+"-th second")

  writer.close()
  print("Video writer closed")