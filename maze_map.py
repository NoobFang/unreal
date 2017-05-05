# record the map of a maze

import numpy as np
from skimage import transform

ACT_LEFT = 0
ACT_RIGHT = 1
ACT_FORWARD = 2
ANGLE_FACTOR = 0.00112294 # 2*pi/(105.5717*53)
FORWARD_FACTOR = 0.003125 # 1 / 320

class MazeMap(object):
  def __init__(self, height=200, width=200):
    self.height = height
    self.width = width
    self.maze = np.zeros((self.height, self.width))
    self.current_h = self.height / 2 # current height position
    self.current_w = self.width / 2 # current width position
    self.current_a = 0.0 # current angle

  def update(self, vtrans, vrot):
    self.current_a -= vrot[1] * ANGLE_FACTOR
    self.current_a = np.fmod(self.current_a, 2*np.pi)
    self.current_h += (FORWARD_FACTOR*vtrans[0])*np.cos(self.current_a) - (FORWARD_FACTOR*vtrans[1])*np.cos(self.current_a+np.pi/2)
    self.current_w += (FORWARD_FACTOR*vtrans[0])*np.sin(self.current_a) - (FORWARD_FACTOR*vtrans[1])*np.sin(self.current_a+np.pi/2)
    h = int(np.floor(self.current_h))
    w = int(np.floor(self.current_w))
    #self.maze[h,w] = min([self.maze[h,w]+0.25, 1.0])
    self.maze[h,w] = 1.0
    # update the maze map if neccessary
    if (self.current_h > self.height - 3):
      self.height += 7
      new_maze = np.zeros((self.height, self.width))
      new_maze[:-7,:] = self.maze
      self.maze = new_maze
    if (self.current_h < 3):
      self.height += 7
      new_maze = np.zeros((self.height, self.width))
      new_maze[7:,:] = self.maze
      self.maze = new_maze
    if (self.current_w >self.width - 3):
      self.width += 7
      new_maze = np.zeros((self.height, self.width))
      new_maze[:,:-7] = self.maze
      self.maze = new_maze
    if (self.current_w < 3):
      self.width += 7
      new_maze = np.zeros((self.height, self.width))
      new_maze[:,7:] = self.maze
      self.maze = new_maze

  def reset(self):
    self.height = 200
    self.width = 200
    self.maze = np.zeros((self.height, self.width))
    self.current_h = self.height / 2 # current height position
    self.current_w = self.width / 2 # current width position
    self.current_a = 0.0 # current angle

  def get_map(self, height, width):
    idx_h, _ = np.nonzero(self.maze)
    if len(idx_h) == 0:
      return transform.resize(self.maze, (height, width))
    min_h = np.min(idx_h)
    max_h = np.max(idx_h)
    idx_w, _ = np.nonzero(np.transpose(self.maze))
    min_w = np.min(idx_w)
    max_w = np.max(idx_w)
    return transform.resize(self.maze[min_h:max_h+1, min_w:max_w+1], (height, width))
    

