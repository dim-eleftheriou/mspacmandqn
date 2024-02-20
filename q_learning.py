import torch
import torch.nn as nn
import torch.nn.functional as F

class Qnetwork(nn.Module):
  """ The architecture uses a CNN network to create emdeddings from images (encoding).
  The architecture of the network expects as input stacked frames (input_channels). Default value is 4.
  ELU activation function is used for Convolutional and Dense Layers (not the last one).
  The output is an array representing the qvalues of the current state for each action (number_of_actions).
  """
  def __init__(self, input_channels=4, number_of_actions=9, conv_size=32, kernel_size=5, embedding_size=512):
    super(Qnetwork, self).__init__()
    self.conv1 = nn.Conv2d(in_channels=input_channels, out_channels=conv_size, kernel_size=kernel_size, stride=2, padding=1)
    self.conv2 = nn.Conv2d(conv_size, conv_size, kernel_size=kernel_size, stride=2, padding=1)
    self.conv3 = nn.Conv2d(conv_size, conv_size, kernel_size=kernel_size, stride=2, padding=1)
    self.conv4 = nn.Conv2d(conv_size, conv_size, kernel_size=kernel_size, stride=2, padding=1)
    self.linear1 = nn.Linear(embedding_size, 128)
    self.linear2 = nn.Linear(128, number_of_actions)

  def forward(self, x):
    x = F.normalize(x)
    y = F.elu(self.conv1(x))
    y = F.elu(self.conv2(y))
    y = F.elu(self.conv3(y))
    y = F.elu(self.conv4(y))
    y = y.flatten(start_dim=2)
    y = y.view(y.shape[0], -1, 32)
    y = y.flatten(start_dim=1)
    y = F.elu(self.linear1(y))
    y = self.linear2(y)
    return y

class CrossAttention(nn.Module):
  """Perform cross attention between state embedding and action embeddings.
  NOTE: state_dimension and action_dimension should be the same.
  NOTE: query_dimension and key_dimension should be the same.
  Queries are creted from states and states and values from actions.
    Arguments:
      state_dimension (int): Dimension of state embeddings
      action_dimension (int): Dimension of action embeddings
      query_dimension (int): Dimension of Q matrix
      key_dimension (int): Dimension of K matrix
      value_dimension (int): Dimension of V matrix
    Returns:
      context_vectors (torch.tensor): Attention matrix
  """
  def __init__(self, state_dimension:int, action_dimension:int, query_dimension:int=128, key_dimension:int=128, value_dimension:int=128):
    # Arguments
    self.query_dimension = query_dimension
    self.key_dimension = key_dimension
    self.value_dimension = value_dimension
    # Architecture
    super(CrossAttention, self).__init__()
    self.W_query = nn.Parameter(torch.randn(state_dimension, query_dimension))
    self.W_key   = nn.Parameter(torch.randn(action_dimension, key_dimension))
    self.W_value = nn.Parameter(torch.randn(action_dimension, value_dimension))

  def forward(self, state_embedding, action_embedding):
    queries_state = state_embedding @ self.W_query
    keys_actions = action_embedding @ self.W_key
    values_actions = action_embedding @ self.W_value
    attention_scores = queries_state @ keys_actions.T
    attention_weights = torch.softmax(attention_scores / self.query_dimension**0.5, dim=-1)
    context_vectors = attention_weights @ values_actions
    return context_vectors

class MultiHeadCrossAttentionModel(nn.Module):
  """Performs multi head cross attention between state embedding and action embeddings.
  NOTE: state_dimension and action_dimension should be the same.
  NOTE: query_dimension and key_dimension should be the same.
  Queries are creted from states and states and values from actions.
    Arguments:
      state_dimension (int): Dimension of state embeddings
      action_dimension (int): Dimension of action embeddings
      query_dimension (int): Dimension of Q matrix
      key_dimension (int): Dimension of K matrix
      value_dimension (int): Dimension of V matrix
    Returns:
      context_vectors (torch.tensor): Attention matrix
  """
  def __init__(self, state_dimension:int, action_dimension:int, query_dimension:int=128, key_dimension:int=128, value_dimension:int=128, num_heads:int=6):
    super().__init__()
    self.heads = nn.ModuleList(
        [CrossAttention(state_dimension, action_dimension, query_dimension, key_dimension, value_dimension) for _ in range(num_heads)]
      )
    self.layer_norm = nn.LayerNorm(value_dimension)

  def forward(self, state_embedding, action_embedding):
    torch.cat([self.layer_norm(head(state_embedding, action_embedding)) for head in self.heads], dim=-1)
    return torch.cat([self.layer_norm(head(state_embedding, action_embedding)) for head in self.heads], dim=-1)

class AlternativeQNetwork(nn.Module):
  """
  Calculates the q values for each state by performing cross attention between
  states and actions.
  """
  def __init__(self, number_of_actions=9, cross_attention_embedding_size=128*6):
    super(AlternativeQNetwork, self).__init__()
    self.MHCA = MultiHeadCrossAttentionModel(state_dimension=512,
                                             action_dimension=128,
                                             query_dimension=128,
                                             key_dimension=128,
                                             value_dimension=128,
                                             num_heads=6)
    self.linear1 = nn.Linear(cross_attention_embedding_size, 256)
    self.linear2 = nn.Linear(256, 128)
    self.linear3 = nn.Linear(128, number_of_actions)

  def forward(self, state_embeddings, action_embeddings):
    x = self.MHCA(state_embeddings, action_embeddings)
    x = F.relu(self.linear1(x))
    x = F.relu(self.linear2(x))
    x = self.linear3(x)
    return x