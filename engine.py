import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

class Tensor():

  '''stores data in ndarray'''

  def __init__(self, data: np.ndarray, parents=()):
    self.data = data
    self.grad = np.zeros(data.shape)
    self.parents = parents
    self._grad_fn = lambda : None

  def __repr__(self):
    return f'\ndata: \n{self.data} \ngrad: \n{self.grad}'

  def __add__(self, other):
    ret = Tensor(data=self.data + other.data, parents=(self, other))

    def _grad_fn():
      self.grad += ret.grad
      other.grad += ret.grad
    ret._grad_fn = _grad_fn 

    return ret

  def __sub__(self, other):
    ret = Tensor(data=self.data - other.data, parents=(self, other))

    def _grad_fn():
      self.grad -= ret.grad
      other.grad -= ret.grad
    ret._grad_fn = _grad_fn 

    return ret

  def __matmul__(self, other):
    ret = Tensor(data=self.data @ other.data, parents=(self, other))

    def _grad_fn():
      self.grad += ret.grad @ other.data.T
      other.grad += self.data.T @ ret.grad
    ret._grad_fn = _grad_fn

    return ret

  def relu(self):
    ret = Tensor(data=np.maximum(self.data, 0), parents=(self,))

    def _grad_fn():
      self.grad += np.multiply(ret.grad, np.where(self.data > 0, 1, 0))
    ret._grad_fn = _grad_fn

    return ret

  def conv(self, kernel, stride=1):
    in_channels, in_height, in_width = self.data.shape
    out_channels, _, kernel_height, kernel_width = kernel.data.shape

    out_height = (in_height - kernel_height) // stride + 1
    out_width = (in_width - kernel_width) // stride + 1

    output = np.zeros((out_channels, out_height, out_width))

    for out_c in range(out_channels):
      channel_output = np.zeros((out_height, out_width))
      for in_c in range(in_channels):
        windows = sliding_window_view(self.data[in_c], (kernel_height, kernel_width))[::stride, ::stride]
        channel_output += np.sum(windows * kernel.data[out_c, in_c, :, :], axis=(2, 3))
      output[out_c] = channel_output

    ret = Tensor(data=output, parents=(self, kernel))

    def _grad_fn():
      for out_c in range(out_channels):
        for in_c in range(in_channels):
          self.grad[in_c] += np.kron(ret.grad[out_c], kernel.data[out_c, in_c])[:in_height, :in_width]
          
          windows = sliding_window_view(self.data[in_c], (kernel_height, kernel_width))[::stride, ::stride]
          kernel.grad[out_c, in_c] = np.sum(windows * ret.grad[out_c, :, :, np.newaxis, np.newaxis], axis=(0, 1))

    ret._grad_fn = _grad_fn

    return ret
  
  def max_pool(self, pool_size, stride):
    channels, height, width = self.data.shape
    
    out_height = (height - pool_size) // stride + 1
    out_width = (width - pool_size) // stride + 1

    output = np.zeros((channels, out_height, out_width))
    indices = np.zeros((channels, out_height, out_width, 2), dtype=int)

    for c in range(channels):
      windows = sliding_window_view(self.data[c], (pool_size, pool_size))[::stride, ::stride]
      
      output[c] = np.max(windows, axis=(2, 3))
      
      max_indices = np.argmax(windows.reshape(out_height, out_width, -1), axis=2)
      indices[c, :, :, 0] = max_indices // pool_size + np.arange(0, height - pool_size + 1, stride)[:, np.newaxis]
      indices[c, :, :, 1] = max_indices % pool_size + np.arange(0, width - pool_size + 1, stride)[np.newaxis, :]

    ret = Tensor(data=output, parents=(self,))

    def _grad_fn():
      self.grad = np.zeros_like(self.data)
      for c in range(channels):
        np.add.at(self.grad[c], 
                  (indices[c, :, :, 0].flatten(), indices[c, :, :, 1].flatten()),
                  ret.grad[c].flatten())

    ret._grad_fn = _grad_fn

    return ret
  
  def tflatten(self):
    ret = Tensor(data=self.data.flatten().reshape(1, -1), parents=(self,))

    def _grad_fn():
      self.grad += ret.grad.reshape(self.data.shape)
    ret._grad_fn = _grad_fn

    return ret
  
  def backward(self, grad):
    self.grad = grad

    topo = []
    visited = set()
    def topo_sort(tensor):
      if tensor not in visited:
        visited.add(tensor)
        for parent in tensor.parents:
          topo_sort(parent)
        topo.append(tensor)

    topo_sort(self)
    for v in reversed(topo):
      v._grad_fn()

  def zero_grad(self):
    self.grad.fill(0)