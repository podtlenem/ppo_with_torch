import torch
from torch import nn
import torch.nn.functional as F
import numpy as np

class NeuralNet(nn.Module):
	def __init__(self, in_dim:int, out_dim:int,hidden_dim:int = 64):
		super(NeuralNet, self).__init__()

		self.layer1 = nn.Linear(in_dim, hidden_dim)
		self.layer2 = nn.Linear(hidden_dim, hidden_dim)
		self.layer3 = nn.Linear(hidden_dim, out_dim)

	def forward(self, obs):
		if isinstance(obs, np.ndarray):
			obs = torch.tensor(obs, dtype=torch.float)

		activation1 = F.relu(self.layer1(obs))
		activation2 = F.relu(self.layer2(activation1))
		output = self.layer3(activation2)

		return output
