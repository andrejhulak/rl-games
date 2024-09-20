import pygame, sys, random

class PongGame:
  def __init__(self, width=1280, height=720):
    pygame.init()
    self.WIDTH, self.HEIGHT = width, height
    self.SCREEN = pygame.display.set_mode((self.WIDTH, self.HEIGHT))
    pygame.display.set_caption("Pong!")
        
    self.FONT = pygame.font.SysFont("Consolas", int(self.WIDTH / 20))
    self.CLOCK = pygame.time.Clock()

    self.right_paddle = pygame.Rect(0, 0, 10, 100)
    self.right_paddle.center = (self.WIDTH - 100, self.HEIGHT / 2)

    self.left_paddle = pygame.Rect(0, 0, 10, 100)
    self.left_paddle.center = (100, self.HEIGHT / 2)

    self.ball = pygame.Rect(0, 0, 20, 20)
    self.ball.center = (self.WIDTH / 2, self.HEIGHT / 2)

    self.right_score = 0
    self.left_score = 0

    self.x_speed, self.y_speed = random.choice([1, -1]), random.choice([1, -1])

  # def handle_input(self):
  #   keys_pressed = pygame.key.get_pressed()
  #   if keys_pressed[pygame.K_UP]:
  #     if self.right_paddle.top > 0:
  #       self.right_paddle.top -= 3
  #   if keys_pressed[pygame.K_DOWN]:
  #     if self.right_paddle.bottom < self.HEIGHT:
  #       self.right_paddle.bottom += 3

  def update_ball(self):
    if self.ball.y >= self.HEIGHT:
      self.y_speed = -1
    if self.ball.y <= 0:
      self.y_speed = 1
    if self.ball.x <= 0:
      self.right_score += 1
      self.reset_game()
    if self.ball.x >= self.WIDTH:
      self.left_score += 1
      self.reset_game()
    if self.right_paddle.x - self.ball.width <= self.ball.x <= self.right_paddle.right and \
      self.right_paddle.top - self.ball.width <= self.ball.y <= self.right_paddle.bottom + self.ball.width:
      self.x_speed = -1
    if self.left_paddle.x - self.ball.width <= self.ball.x <= self.left_paddle.right and \
      self.left_paddle.top - self.ball.width <= self.ball.y <= self.left_paddle.bottom + self.ball.width:
      self.x_speed = 1

    self.ball.x += self.x_speed * 2
    self.ball.y += self.y_speed * 2

  def reset_game(self):
    self.ball.center = (self.WIDTH / 2, self.HEIGHT / 2)
    self.x_speed, self.y_speed = random.choice([1, -1]), random.choice([1, -1])

    self.right_paddle.center = (self.WIDTH - 100, self.HEIGHT / 2)
    self.left_paddle.center = (100, self.HEIGHT / 2)

  def update_left_paddle(self, input_y):
    self.left_paddle.bottom += input_y

  def update_right_paddle(self):
    if self.right_paddle.y < self.ball.y:
      self.right_paddle.top += 1
    if self.right_paddle.bottom > self.ball.y:
      self.right_paddle.bottom -= 1

  def draw(self):
    self.SCREEN.fill("black")
    pygame.draw.rect(self.SCREEN, "white", self.right_paddle)
    pygame.draw.rect(self.SCREEN, "white", self.left_paddle)
    pygame.draw.circle(self.SCREEN, "white", self.ball.center, 10)

    right_paddle_score_text = self.FONT.render(str(self.right_score), True, "white")
    left_paddle_score_text = self.FONT.render(str(self.left_score), True, "white")

    self.SCREEN.blit(right_paddle_score_text, (self.WIDTH / 2 + 50, 50))
    self.SCREEN.blit(left_paddle_score_text, (self.WIDTH / 2 - 50, 50))

  def run(self, input_y):
    for event in pygame.event.get():
      if event.type == pygame.QUIT:
        pygame.quit()
        sys.exit()

    # self.handle_input()
    self.update_ball()
    self.update_left_paddle(input_y)
    self.update_right_paddle()
    self.draw()

    pygame.display.update()
    self.CLOCK.tick(100000)