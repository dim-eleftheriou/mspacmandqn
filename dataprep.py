import numpy as np
import torch

class DataPrep:
  def __init__(self):
    pass

  # Preprocess image
  def preprocess(image:np.array) -> np.array:
    # Crop the image
    image = image[:172, :]
    # Downsample by factor of 2
    image = image[::2,::2]
    return image.astype(np.float32) / 255.0

  @classmethod
  def prepare_state(cls, state:np.array) -> np.array:
    return torch.from_numpy(cls.preprocess(state)).float().unsqueeze(dim=0)

  @classmethod
  def prepare_initial_state(cls, state:np.array, N:int=4) -> np.array:
    """This function is used for creating the initial state of the environment.
    All frames of the initial state will be identical"""
    state_ = torch.from_numpy(cls.preprocess(state)).float()
    tmp = state_.repeat((N, 1, 1))
    return tmp.unsqueeze(dim=0)

  # @classmethod
  # def prepare_multi_state(cls, state1:torch.Tensor, state2:np.array) -> np.array:
  #   """This function is used for stacking the new frame into current state.
  #   It will be used only in testing phase of the model"""
  #   state1 = state1.clone()
  #   tmp = torch.from_numpy(cls.preprocess(state2)).float()
  #   state1[0][0] = state1[0][1]
  #   state1[0][1] = state1[0][2]
  #   state1[0][2] = state1[0][3]
  #   state1[0][3] = tmp
  #   return state1
  
  @classmethod
  def prepare_multi_state(cls, state1:torch.Tensor, state2:np.array) -> np.array:
    """This function is used for stacking the new frame into current state.
    It will be used only in testing phase of the model"""
    state = state1.clone()
    tmp = torch.from_numpy(cls.preprocess(state2)).float()
    state[0] = state1[1]
    state[1] = state1[2]
    state[2] = state1[3]
    state[3] = tmp
    return state