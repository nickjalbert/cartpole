# Thin wrapper around cartpole from OpenAI's Gym toolkit
# This env models a cart with a pole balancing on top of it
import agentos
import gym


class CartPole(agentos.Environment):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.cartpole = gym.make('CartPole-v1')
        self.reset()

    def step(self, action):
        assert action in [0, 1]
        result = self.cartpole.step(action)
        self.last_obs, self.last_reward, self.done, self.info = result
        return result

    @property
    def valid_actions(self):
        return [0, 1]

    def reset(self):
        self.last_obs = None 
        self.last_reward = None
        self.last_done = False
        self.last_info = None
        self.last_obs = self.cartpole.reset()
        return self.last_obs



# Unit tests for Corridor
def run_tests():
    print("Testing Cartpole...")
    env = CartPole()
    first_obs = env.reset()
    assert len(first_obs) == 4
    obs, reward, done, info = env.step(0)
    obs, reward, done, info = env.step(1)
    while not done:
        obs, reward, done, info = env.step(1)
    print("Test successful...")

if __name__ == "__main__":
    run_tests()
