from pong import PongGame
import engine, nn, optimizer
import numpy as np

W = 1280
H = 720

def get_nn_in(game):
  return [game.ball.x / W, game.ball.y / H, game.left_paddle.y / H, game.right_paddle.y / H]

def softmax(x):
  x_max = np.max(x, axis=-1, keepdims=True)
  exp_x = np.exp(x - x_max)
  return exp_x / np.sum(exp_x, axis=-1, keepdims=True)

def nn_out_to_y(out):
  probs = softmax(out.data).reshape(2)
  # 0 is up, 1 is down
  choice = np.random.choice([0, 1], p=probs)
  # up
  # if choice == 0:
  #   ret = -2
  # # down
  # if choice == 1:
  #   ret = 2
  return choice

if __name__ == "__main__":
  game = PongGame(W, H)

  # TODO: maybe add ball speed?
  in_shape = 4
  hidden_layers = 2
  hidden_units = 16
  # output shape = 2 because we have prob of going up and down
  out_shape = 2

  ai = nn.NeuralNet(input_shape=in_shape, hidden_layers=hidden_layers, hidden_units=hidden_units, output_shape=out_shape) 

  optim = optimizer.Optimizer(parameters=ai.parameters)

  input_y = 0
  list_out = []
  list_actions = []
  input_tensors = []
  win = False
  lose = False
  prev_left_score = 0
  prev_right_score = 0

  while True:
    game.run(input_y)

    if prev_left_score != game.left_score:
      win = True
    else:
      win = False

    if prev_right_score != game.right_score:
      lose = True
    else:
      lose = False

    input_tensor = engine.Tensor(data=np.array(get_nn_in(game)))
    input_tensors.append(input_tensor)

    out = ai.forward(input_tensor)
    list_out.append(out)
    
    choice = nn_out_to_y(out)
    list_actions.append(choice)

    if choice == 0:
      input_y = -2
    if choice == 1:
      input_y = 2

    if win == True:
      for i in range(len(list_out)):
        grad = np.zeros(out_shape)
        grad[list_actions[i]] = 1
        grad = grad.reshape(1, -1)
        print(grad.shape)
        print(grad)
        list_out[i].backward(grad=grad)

      optim.optimize(learning_rate=0.001)

      ai.forget_inter()
      input_tensors = []
      win = False

    if lose == True:
      for i in range(len(list_out)):
        grad = np.zeros(out_shape)
        grad[list_actions[i]] = 1
        grad = grad.reshape(1, -1)
        print(grad.shape)
        print(grad)
        list_out[i].backward(grad=-grad)

      optim.optimize(learning_rate=0.001)

      ai.forget_inter()
      input_tensors = []
      lose = False

    prev_left_score = game.left_score
    prev_right_score = game.right_score