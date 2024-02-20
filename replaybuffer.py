import torch
import numpy as np
from random import shuffle

class ExperienceReplay:
  """ExperienceReplay class is used for storing the trajectories over time.
  Arguments for instatiating the class:
    N: Maximum number of trajectories stored in memory list. Default value is 500
    batch_size: Number of trajectories to sample from memory list. Default value is 100
    """
  def __init__(self, N:int=5000, batch_size:int=128):
    self.N = N
    self.batch_size = batch_size
    self.memory = []
    self.counter = 0

  # Append memory list with experience tuples
  def add_memory(self, state1, action, reward, state2, terminated):
    self.counter +=1
    # Shuffle memory when limit is reached
    if self.counter % self.N == 0:
      self.shuffle_memory()
    # Add a new trajectory if there is capacity
    if len(self.memory) < self.N:
      self.memory.append( (state1, action, reward, state2, terminated) )
    # Replace randomly an old trajectory with a new one if capacity is reached
    else:
      rand_index = np.random.randint(0, self.N-1)
      self.memory[rand_index] = (state1, action, reward, state2, terminated)

  # Shuffle the trajectories in memory
  def shuffle_memory(self):
    shuffle(self.memory)

  # Create batches of trajectories
  def get_batch(self):
    # Define batch size
    if len(self.memory) < self.batch_size:
      batch_size = len(self.memory)
    else:
      batch_size = self.batch_size
    if len(self.memory) < 1:
      print("Error: No data in memory.")
      return None
    # Select randomly trajectories from memory
    ind = np.random.choice(np.arange(len(self.memory)), batch_size, replace=False)
    batch = [self.memory[i] for i in ind] #batch is a list of tuples (state1, action, reward, state2, terminated)
    state1_batch = torch.stack([x[0].squeeze(dim=0) for x in batch], dim=0)
    action_batch = torch.tensor([x[1] for x in batch]).long()
    reward_batch = torch.Tensor([x[2] for x in batch])
    state2_batch = torch.stack([x[3].squeeze(dim=0) for x in batch], dim=0)
    terminated_batch = torch.Tensor([x[4] for x in batch])
    return state1_batch, action_batch, reward_batch, state2_batch, terminated_batch