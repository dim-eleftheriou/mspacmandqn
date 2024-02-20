import torch
import torch.nn as nn
import torch.nn.functional as F

class Encoder(nn.Module):
  """
  The encoder model takes a stacked-frames state representation
  and encodes it in embeddings of size embedding_size.
  """
  def __init__(self, input_channels=4, conv_size=32, kernel_size=5, embedding_size=512):
    self.embedding_size = embedding_size
    super(Encoder, self).__init__()
    self.conv1 = nn.Conv2d(input_channels, conv_size, kernel_size=kernel_size, stride=2, padding=1)
    self.conv2 = nn.Conv2d(conv_size, conv_size, kernel_size=kernel_size, stride=2, padding=1)
    self.conv3 = nn.Conv2d(conv_size, conv_size, kernel_size=kernel_size, stride=2, padding=1)
    self.conv4 = nn.Conv2d(conv_size, conv_size, kernel_size=kernel_size, stride=2, padding=1)

  def forward(self, x):
      x = F.normalize(x)
      y = F.elu(self.conv1(x))
      y = F.elu(self.conv2(y))
      y = F.elu(self.conv3(y))
      y = F.elu(self.conv4(y))
      y = y.flatten(start_dim=1) #size N, embedding_size
      return y

class InverseModel(nn.Module):
  """
  The inverse model takes as input two ENCODED consecutive states S[t] and S[t+1] (embeddings)
  and predicts the action that led the agent to go from S[t] to S[t+1].
  Action returned is a vector with shape number_of_actions and contains the logits of each action.
  """
  def __init__(self, encoder_embedding_size= 512, number_of_actions=9):
    super(InverseModel, self).__init__()
    self.linear1 = nn.Linear(2*encoder_embedding_size, 256)
    self.linear2 = nn.Linear(256, number_of_actions)

  def forward(self, state1, state2):
    x = torch.cat( (state1, state2) , dim=1)
    y = F.relu(self.linear1(x))
    y = self.linear2(y)
    #y = F.softmax(y, dim=1)
    return y

class ForwardModel(nn.Module):
  """
  The forward model takes as input an encoded state S[t] and the action taken in that state a[t] as
  and predicts the encoded state S[t+1]. Action is either one-hot encoded representation or 
  embedding representation, so it has shape equal to number_of_actions.
  """
  def __init__(self, encoder_embedding_size=512, number_of_actions=128):
    self.number_of_actions = number_of_actions
    super(ForwardModel, self).__init__()
    self.linear1 = nn.Linear(encoder_embedding_size + number_of_actions, encoder_embedding_size)
    self.linear2 = nn.Linear(encoder_embedding_size, encoder_embedding_size)

  def forward(self, state, action):
    x = torch.cat( (state, action) ,dim=1)
    y = F.relu(self.linear1(x))
    y = self.linear2(y)
    return y