# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from multiprocessing import Process, Pipe
import cv2
import numpy as np
import deepmind_lab

from environment import environment
from constants import ENV_NAME

COMMAND_RESET     = 0
COMMAND_ACTION    = 1
COMMAND_TERMINATE = 2

def worker(conn):
  level = ENV_NAME
  env = deepmind_lab.Lab(
    level,
    ['RGBD_INTERLACED', 'VEL.TRANS', 'VEL.ROT'], # return RGBD, velocity of transport, velocity of rotation
    config={
      'fps': str(60),
      'width': str(84),
      'height': str(84)
    })
  conn.send(0)
  
  while True:
    command, arg = conn.recv()

    if command == COMMAND_RESET:
      env.reset()
      obs = env.observations()
      conn.send(obs)
    elif command == COMMAND_ACTION:
      reward = env.step(arg, num_steps=4)
      terminal = not env.is_running()
      if not terminal:
        obs = env.observations()
      else:
        obs = 0
      conn.send([obs, reward, terminal])
    elif command == COMMAND_TERMINATE:
      break
    else:
      print("bad command: {}".format(command))
  env.close()      
  conn.send(0)
  conn.close()


def _action(*entries):
  return np.array(entries, dtype=np.intc)


class LabEnvironment(environment.Environment):
  ACTION_LIST = [
    _action(-16,   0,  0,  0, 0, 0, 0), # look_left
    _action( 16,   0,  0,  0, 0, 0, 0), # look_right
    #_action(  0,  10,  0,  0, 0, 0, 0), # look_up
    #_action(  0, -10,  0,  0, 0, 0, 0), # look_down
    #_action(  0,   0, -1,  0, 0, 0, 0), # strafe_left
    #_action(  0,   0,  1,  0, 0, 0, 0), # strafe_right
    _action(  0,   0,  0,  1, 0, 0, 0), # forward
    #_action(  0,   0,  0, -1, 0, 0, 0), # backward
    #_action(  0,   0,  0,  0, 1, 0, 0), # fire
    #_action(  0,   0,  0,  0, 0, 1, 0), # jump
    #_action(  0,   0,  0,  0, 0, 0, 1),  # crouch
    #_action(  0,  0,  0,  0,  0, 0, 0) # noop
  ]

  @staticmethod
  def get_action_size():
    return len(LabEnvironment.ACTION_LIST)
  
  def __init__(self):
    environment.Environment.__init__(self)
    self.depth_bias = np.zeros((84,84,4), dtype=np.uint8)
    self.depth_bias[:,:,3] = np.ones((84,84), dtype=np.uint8)*180
    self.conn, child_conn = Pipe()
    self.proc = Process(target=worker, args=(child_conn,))
    self.proc.Daemon = True
    self.proc.start()
    self.conn.recv()
    self.reset()

  def reset(self):
    self.conn.send([COMMAND_RESET, 0])
    obs = self.conn.recv()

    state = self._preprocess_frame(obs['RGBD_INTERLACED'] - self.depth_bias)
    #self.last_state = np.expand_dims(state[:,:,3], -1)
    self.last_state = state
    self.last_vtrans = obs['VEL.TRANS']
    self.last_vrot = obs['VEL.ROT']
    self.last_action = 0
    self.last_reward = 0

  def stop(self):
    self.conn.send([COMMAND_TERMINATE, 0])
    ret = self.conn.recv()
    self.conn.close()
    self.proc.join()
    print("lab environment stopped")
    
  def _preprocess_frame(self, image):
    image = image.astype(np.float32)
    image = image / 255.0
    return image

  def process(self, action):
    real_action = LabEnvironment.ACTION_LIST[action]

    self.conn.send([COMMAND_ACTION, real_action])
    obs, reward, terminal = self.conn.recv()

    if not terminal:
      state = self._preprocess_frame(obs['RGBD_INTERLACED'] - self.depth_bias)
      #state = np.expand_dims(state[:,:,3], -1)
      vtrans = obs['VEL.TRANS']
      vrot = obs['VEL.ROT']
    else:
      state = self.last_state
      vtrans = self.last_vtrans
      vrot = self.last_vrot
    
    pixel_change = self._calc_pixel_change(state, self.last_state)
    self.last_state = state
    self.last_vtrans = vtrans
    self.last_vrot = vrot
    self.last_action = action
    self.last_reward = reward
    return state, reward, terminal, pixel_change, vtrans, vrot
