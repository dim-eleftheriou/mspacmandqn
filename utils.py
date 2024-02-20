import torch
from math import log
import torch.nn.functional as F

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