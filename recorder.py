from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import pickle as pkl
import cv2
import pygame, sys, os
from pygame.locals import *
import scipy.misc as misc

from environment.environment import Environment
from train.experience import Experience, ExperienceFrame
from constants import *

MAX_EXP = 1e4
DISP_SIZE = (84, 84) # width, height
FPS = 15
BLACK = (0, 0, 0)

class Recorder(object):
  """
  Record the actions played by human to accelerate the reinforcement learning
  """
  def __init__(self):
    self.env = Environment.create_environment()
    if os.path.exists('human_exp.pkl'):
      with open('human_exp.pkl', 'r') as f:
        self.ExpPool = pkl.load(f)
    else:
      self.ExpPool = Experience(MAX_EXP)
    pygame.init()
    self.surface = pygame.display.set_mode(DISP_SIZE, 0)
    pygame.display.set_caption('Recorder')

  def update(self):
    self.surface.fill(BLACK)
    obs, reward, terminal, pc, action = self.process()
    if action != 3:
      self.record(obs, reward, terminal, pc, action)
    pygame.display.update()

  def choose_action(self):
    action = 3
    pressed = pygame.key.get_pressed()
    if pressed[pygame.K_a]:
      action = 0
    elif pressed[pygame.K_d]:
      action = 1
    elif pressed[pygame.K_w]:
      action = 2
    return action

  def process(self):
    action = self.choose_action()
    obs, reward, terminal, pc = self.env.process(action)
    #data = misc.imresize(obs*255.0, DISP_SIZE)
    data = obs * 255.0
    image = pygame.image.frombuffer(data.astype(np.uint8), DISP_SIZE, "RGB")
    self.surface.blit(image, (0,0))
    if terminal:
      self.env.reset()
    return obs, reward, terminal, pc, action

  def record(self, obs, reward, terminal, pc, action):
    last_state = self.env.last_state
    last_action = self.env.last_action
    last_reward = self.env.last_reward
    frame = ExperienceFrame(last_state, reward, action, terminal, pc, last_action, last_reward)
    self.ExpPool.add_frame(frame)
    if self.ExpPool.is_full():
      print('Experience pool is filled!')
    print('Filled %d/%d.' % (len(self.ExpPool._frames), MAX_EXP), end='\r')
    sys.stdout.flush()
    

  def save(self):
    with open('human_exp.pkl', 'w') as f:
      pkl.dump(self.ExpPool, f)


if __name__ == '__main__':
  recorder = Recorder()
  clock = pygame.time.Clock()
  running = True
  while running:
    for event in pygame.event.get():
      if event.type == pygame.QUIT:
        running = False
        recorder.save()

    recorder.update()
    clock.tick(FPS)
