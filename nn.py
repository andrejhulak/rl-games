import numpy as np
from random import uniform
from engine import *

inter = []

def init_radom_weights(shape1, shape2):
  bound = np.sqrt(6 / (shape1 + shape2))
  return Tensor(np.random.uniform(-bound, bound, (shape1, shape2)))

class NeuralNet(object):
    def __init__(self, input_shape, hidden_layers, hidden_units, output_shape):
      self.parameters = []

      self.input_layer = init_radom_weights(input_shape, hidden_units)
      self.parameters.append(self.input_layer)

      self.hidden_layers_dict = {}
      for i in range(hidden_layers):
        self.hidden_layers_dict[f'hidden_layer_{i}'] = init_radom_weights(hidden_units, hidden_units)
        self.parameters.append(self.hidden_layers_dict[f'hidden_layer_{i}'])

      self.output_layer = init_radom_weights(hidden_units, output_shape)
      self.parameters.append(self.output_layer)

      self.bias_vec = [Tensor(np.zeros((1, hidden_units))) for _ in range(hidden_layers)]
      self.bias_vec.append(Tensor(np.zeros((1, output_shape))))

      for i in range(len(self.bias_vec)):
        self.parameters.append(self.bias_vec[i])

    def forward(self, x):
      inter.append(x)

      x = x @ self.input_layer
      inter.append(x)

      x = x.relu()
      inter.append(x)

      for i in range(len(self.hidden_layers_dict)):
        x = x @ self.hidden_layers_dict[f'hidden_layer_{i}'] + self.bias_vec[i]
        inter.append(x)
        x = x.relu()
        inter.append(x)

      x = x @ self.output_layer + self.bias_vec[-1]
      inter.append(x)

      return x     

    def forget_inter(self):
      global inter
      inter = []

    def print_weights(self):
      # print(self.input_layer)
      #  for i in range(len(self.hidden_layers_dict)):
        # print(self.hidden_layers_dict[f'hidden_layer_{i}'])
      print(self.output_layer.data)