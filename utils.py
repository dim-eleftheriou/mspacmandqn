import torch
from math import log
import torch.nn.functional as F
from tqdm import tqdm
from mspacmandqn.dataprep import DataPrep
import numpy as np

def policy(qvalues:torch.tensor, number_of_actions:int=9, eps:float=None) -> int:
  """This function creates an epsilon greedy policy if epsilon is provided.
  Otherwise randomly selects an action using multinomial distribution, by
  passing probabilities of each class calculated as softmax of normalized qvalues.
  Normalization of q values smooths the probabilities"""
  if eps is not None:
    # Select a random action with probability epsilon
    if torch.rand(1) < eps:
      return torch.randint(low=0, high=number_of_actions, size=(1, ))
    # Select the action with the highest q value with probability 1 - epsilon
    else:
      return torch.argmax(qvalues)
  else:
      return torch.multinomial(F.softmax(F.normalize(qvalues)), num_samples=1)

def transform_reward(reward, base=1000):
    """Transform the reward apllying the logarithm function"""
    return log(reward, base) if reward > 0 else reward

# Calculate width/height size after a CONV2D layer
def calculate_size(input_size:int, kernel_size:int, stride:int, padding:int)->int:
  return int((input_size - kernel_size + 2*padding)/stride + 1)

# Calculate width/height size after multiple CONV2D layers with same arguments
def final_size(calculate_size, repeat:int, args:tuple) -> int:
  size = calculate_size(*args)
  for i in range(repeat-1):
    n_args = tuple([size, *args[1:]])
    size = calculate_size(*n_args)
  return size

def random_play(env, num_of_games):
    """
    Play the mspacman with totaly random actions.
      Arguments:
        env: the environment to play
        num_of_games (int): How many games should be played
      Returns:
        scores (list): List with scores achived in games
    """
    scores = []
    for game in tqdm(range(num_of_games), desc="Playing…", ascii=False, ncols=75):
      observation = env.reset()
      score = 0
      terminated = False
      while not terminated:
        # Random action
        action = env.action_space.sample()
        # Run one timestep
        observation, reward, terminated, truncated, info = env.step(action)
        score += reward
      scores.append(score)
    return scores


def test_the_agent(env, agent, num_of_games):
    """
    Play the mspacman using an agent.
      Arguments:
        env: the environment to play
        agent: an agent class
        num_of_games (int): How many games should be played
      Returns:
        scores (list): List with scores achived in games
    """
    scores = []
    for game in tqdm(range(num_of_games), desc="Playing…", ascii=False, ncols=75):
      score = 0
      observation = env.reset()
      state = DataPrep.prepare_initial_state(observation).to(agent.DEVICE)
      terminated = False
      while not terminated:
        action = agent.act(state)
        next_state, reward, terminated, truncated, info = env.step(action)
        next_state = DataPrep.prepare_multi_state(state, next_state).to(agent.DEVICE)
        state = next_state
        score += reward
      scores.append(score)
    print(f"Average score of the Agent is: {np.mean(scores)}")
    return scores