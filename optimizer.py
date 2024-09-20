import numpy as np
from engine import *

class Optimizer():
  def __init__(self, parameters):
    self.parameters = parameters

  def optimize(self, learning_rate):
    for param in self.parameters:
      param.data -= learning_rate * param.grad
  
  def zero_grad(self):
    for param in self.parameters:
      param.zero_grad()