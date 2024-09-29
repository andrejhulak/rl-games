from pong import PongGame
import engine, nn, optimizer
import numpy as np
from scipy.special import expit

W = 1280
H = 720

def get_nn_in(game):
  return [game.ball.x / W, game.ball.y / H, game.left_paddle.y / H, game.right_paddle.y / H, game.x_speed / W, game.y_speed/ H]
  # return [game.ball.x, game.ball.y, game.left_paddle.y, game.right_paddle.y]

def softmax(x):
  x_max = np.max(x, axis=-1, keepdims=True)
  exp_x = np.exp(x - x_max)
  return exp_x / np.sum(exp_x, axis=-1, keepdims=True)

def sigmoid(x): 
  return 1.0 / (1.0 + np.exp(-x))

# def nn_out_to_y(out):
#   # print(out)
#   # probs = softmax(out.data).reshape(2)
#   # print(f'Softmax: {probs} and {sum(probs)}')
#   probs = sigmoid(out.data).reshape(2)
#   probs /= sum(probs)
#   # print(f'Sigmoid: {probs} and {sum(probs)}')
#   # 0 is up, 1 is down
#   choice = np.random.choice([0, 1], p=probs)
#   # choice = np.argmax(probs, axis=0)
#   return choice

def nn_out_to_y(out):
  logits = out.data.reshape(2)
  probs = softmax(logits)
  print(probs)
  log_probs = np.log(probs)
  choice = np.random.choice([0, 1], p=probs)
  return choice, log_probs[choice]

def calculate_reward_and_action(game):
    paddle_center = game.left_paddle.y + game.left_paddle.height / 2
    ball_y = game.ball.y
    
    if paddle_center > ball_y:
        return 1.0, 0  
    elif paddle_center < ball_y:
        return 1.0, 1 
    else:
        return 2.0, None  

if __name__ == "__main__":
  game = PongGame(W, H)

  lr = 0.0001

  # TODO: maybe add ball speed?
  in_shape = 6
  hidden_layers = 4
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

  i = 0

  while True:
    game.run(input_y)

    reward, correct_action = calculate_reward_and_action(game)

    input_tensor = engine.Tensor(data=np.array(get_nn_in(game)).reshape(1, -1))
    input_tensors.append(input_tensor)

    out = ai.forward(input_tensor)
    list_out.append(out)
    
    choice, log_prob = nn_out_to_y(out)
    list_actions.append((choice, log_prob))
    
    if correct_action is not None:
      grad = np.zeros(out_shape)
      grad[correct_action] = 1.0
      grad[1 - correct_action] = -1.0 
      grad = grad.reshape(1, -1)
      out.backward(grad=grad)

    if choice == 0:
      input_y = -3  
    else:
      input_y = 3 

    # print(f"Reward: {grad}, Correct Action: {correct_action}, Chosen Action: {choice}")
# 
    # if len(list_out) >= 10:  
    if prev_game_state['right_score'] < game.right_score:
      optim.optimize(learning_rate=lr)
      optim.zero_grad()

      ai.forget_inter()
      input_tensors = []
      list_actions = []
      list_out = []