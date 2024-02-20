import torch
import torch.nn as nn

class ActionNetwork(nn.Module):
  """
  Creates action embeddings of size 128 embedding_size.
  """
  def __init__(self, number_of_actions=9, embedding_size=128):
    super(ActionNetwork, self).__init__()
    self.embedding_matrix = nn.Embedding(number_of_actions, embedding_size)

  def forward(self, x):
    # Calculate embedding matrix using actions
    embedding_matrix = self.embedding_matrix(x)
    # Find which actions where selected in the batch of one hot encoded actions
    actions = [y.nonzero().item() for y in x]
    action_embeddings = [embedding_matrix[x][y] for x, y in enumerate(actions)]
    return torch.stack(action_embeddings)