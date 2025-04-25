
import torch.nn as nn
import torch



class WorldNet(nn.Module):
    def __init__(self, actor_network,device):
        super(WorldNet, self).__init__()
        self.STATE_DIM = 6
        self.N_STEPS = 100
        self.device = device
        self.actor_network = actor_network

        # Take state and action and ouput new state
        self.f = torch.nn.Sequential(nn.Linear(8, 16, bias=True),
                                     nn.ReLU6(),
                                     nn.Linear(16, 32, bias=True),
                                     nn.ReLU6(),
                                     nn.Linear(32, 16, bias=True),
                                     nn.ReLU6(),
                                     nn.Linear(16, 8, bias=True),
                                     nn.ReLU6(),
                                     nn.Linear(8, 6, bias=True),
                                     )
        self.mlp_reward = nn.Linear(6, 2, bias=True)

    def forward(self, u):
        '''
        :param u: batch_size  x 6+2 (batch of initial_states)
        :return: Sequence of states and rewards
        '''
        # Initial state (position, velocity, goal)
        x = torch.zeros(u.shape[0], 1, self.STATE_DIM).to(self.device)
        x[:, 0, :] = u

        for t in range(1, self.N_STEPS + 1):
            x_new = x[:, t - 1, :] + self.f(torch.hstack([x[:, t - 1, :], self.actor_network(x[:, t - 1, :])]))
            x = torch.cat((x, x_new.unsqueeze_(1)), 1)

        # Calculate reward for each episode and time.
        reward_prediction = torch.exp(-torch.sqrt(torch.sum(self.mlp_reward(x[:, 1:, :]) ** 2, dim=2, keepdim=True)))
        return torch.concatenate([x[:, :-1, :], reward_prediction], dim=2)


