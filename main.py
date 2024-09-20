from pong import PongGame
import engine, nn, optimizer
import numpy as np

W = 1280
H = 720

def get_nn_in(game):
  return [game.ball.x / W, game.ball.y / H, game.left_paddle.y / H, game.right_paddle.y / H]
  # return [game.ball.x, game.ball.y, game.left_paddle.y, game.right_paddle.y]

def softmax(x):
  x_max = np.max(x, axis=-1, keepdims=True)
  exp_x = np.exp(x - x_max)
  return exp_x / np.sum(exp_x, axis=-1, keepdims=True)

def nn_out_to_y(out):
  probs = softmax(out.data).reshape(2)
  # 0 is up, 1 is down
  choice = np.random.choice([0, 1], p=probs)
  return choice

def calculate_reward(game, prev_game_state):
    reward = 0
    
    paddle_to_ball_dist = abs(game.left_paddle.y - game.ball.y)
    prev_paddle_to_ball_dist = abs(prev_game_state['paddle_y'] - prev_game_state['ball_y'])
    reward += (prev_paddle_to_ball_dist - paddle_to_ball_dist) * 0.1
    
    if game.ball.x > W/2:
        reward -= abs(game.left_paddle.y - H/2) * 0.05
    
    if game.left_score > prev_game_state['left_score']:
        reward += 10
    if game.right_score > prev_game_state['right_score']:
        reward -= 10
    
    reward -= 0.01
    
    return reward

if __name__ == "__main__":
  game = PongGame(W, H)

  lr = 0.000001

  # TODO: maybe add ball speed?
  in_shape = 4
  hidden_layers = 2 
  hidden_units = 64
  # output shape = 2 because we have prob of going up and down
  out_shape = 2

  ai = nn.NeuralNet(input_shape=in_shape, hidden_layers=hidden_layers, hidden_units=hidden_units, output_shape=out_shape) 

  optim = optimizer.Optimizer(parameters=ai.parameters)

  input_y = 0
  list_out = []
  list_actions = []
  input_tensors = []
  prev_game_state = {
      'ball_x': game.ball.x,
      'ball_y': game.ball.y,
      'paddle_y': game.left_paddle.y,
      'left_score': game.left_score,
      'right_score': game.right_score
  }

  while True:
    game.run(input_y)

    reward = calculate_reward(game, prev_game_state)

    input_tensor = engine.Tensor(data=np.array(get_nn_in(game)).reshape(1, -1))
    input_tensors.append(input_tensor)

    out = ai.forward(input_tensor)
    list_out.append(out)
    
    choice = nn_out_to_y(out)
    list_actions.append(choice)

    if choice == 0:
      input_y = -3
    if choice == 1:
      input_y = 3

    grad = np.zeros(out_shape)
    grad[choice] = reward
    grad = grad.reshape(1, -1)
    out.backward(grad=grad)

    if len(list_out) >= 20:
      optim.optimize(learning_rate=lr)
      optim.zero_grad()

      ai.forget_inter()
      input_tensors = []
      list_actions = []
      list_out = []

    prev_game_state = {
        'ball_x': game.ball.x,
        'ball_y': game.ball.y,
        'paddle_y': game.left_paddle.y,
        'left_score': game.left_score,
        'right_score': game.right_score
    }

    # ai.print_weights()