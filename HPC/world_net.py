import torch
from torch.utils.data import TensorDataset, DataLoader
import gymnasium as gym
import numpy as np
import torch.nn as nn

# %%

device = 'cuda'


class WorldNet(nn.Module):
    def __init__(self, actor_network):
        super(WorldNet, self).__init__()
        self.STATE_DIM = 6
        self.N_STEPS = 100
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
        x = torch.zeros(u.shape[0], 1, self.STATE_DIM).to(device)
        x[:, 0, :] = u

        # State, x, evolves over time. Updates are non-linear functions of the state (p,v,g) and the action.
        for t in range(1, self.N_STEPS + 1):
            x_new = x[:, t - 1, :] + self.f(torch.hstack([x[:, t - 1, :], self.actor_network(x[:, t - 1, :])]))
            x = torch.cat((x, x_new.unsqueeze_(1)), 1)

        # Calculate reward for each episode and time.
        reward_prediction = torch.exp(-torch.sqrt(torch.sum(self.mlp_reward(x[:, 1:, :]) ** 2, dim=2, keepdim=True)))
        return torch.concatenate([x[:, :-1, :], reward_prediction], dim=2)


class ActorNet(nn.Module):
    def __init__(self):
        super(ActorNet, self).__init__()

        self.mlp = nn.Sequential(nn.Linear(6, 8, bias=True),
                                 nn.Tanh(),
                                 nn.Linear(8, 16, bias=True),
                                 nn.Tanh(),
                                 nn.Linear(16, 32, bias=True),
                                 nn.Tanh(),
                                 nn.Linear(32, 16, bias=True),
                                 nn.Tanh(),
                                 nn.Linear(16, 8, bias=True),
                                 nn.Tanh(),
                                 nn.Linear(8, 2, bias=True))

    def forward(self, x):
        return self.mlp(x)


print('generating data')
# Generate data for world model

MEDIUM_MAZE = [
    [1, 1, 1, 1, 1, 1, 1, 1],
    [1, 0, 0, 0, 0, 0, 0, 1],
    [1, 1, 1, 1, 1, 1, 0, 1],
    [1, 1, 1, 1, 1, 1, 'g', 1],
    [1, 1, 1, 1, 1, 1, 0, 1],
    [1, 1, 1, 1, 1, 1, 0, 1],
    [1, 0, 0, 0, 0, 0, 0, 1],
    [1, 1, 1, 1, 1, 1, 1, 1]]

# Initialize environment with this map for training.
env = gym.make('PointMaze_MediumDense-v3',
               max_episode_steps=100, maze_map=MEDIUM_MAZE,
               render_mode='rgb_array'
               )

# Initialize networks
# actor_network = ActorNet().to(device)
world_network = WorldNet(actor_network=ActorNet().to(device)).to(device)


# Creat dataset
def generate_experience(actor, device='cpu'):
    NUM_EPISODES = 1000
    MAX_STEPS = 100
    inputs = torch.zeros(NUM_EPISODES, 6)
    labels = torch.zeros(NUM_EPISODES, MAX_STEPS, 7)
    for episode in range(NUM_EPISODES):
        # Initialize new actor

        # Reset environment
        state = env.reset()
        # Reshape state variable
        state = torch.tensor(np.concatenate([state[0]['observation'], state[0]['desired_goal']])).float()
        inputs[episode, :] = state

        for step in range(MAX_STEPS):
            action = actor(state.to(device))
            # Take step in environment to get new state and reward
            new_state, reward, _, _, _ = env.step(action.detach().cpu())
            new_state = torch.tensor(np.concatenate([new_state['observation'], new_state['desired_goal']])).float()

            # Add data
            labels[episode, step, :] = torch.hstack([state, torch.tensor(reward).float()])

            # Set current state to new state
            state = new_state.clone()

    inputs = inputs.to(device)
    labels = labels.to(device)
    return inputs, labels


NUM_EPOCHS = 50000

for run in range(1):
    # Train world model
    print('Training world net')
    inputs, labels = generate_experience(world_network.actor_network, device=device)
    my_dataset = TensorDataset(inputs, labels)  #
    my_dataloader = DataLoader(my_dataset, batch_size=256, shuffle=True)

    # Turn gradients for actor
    for param in world_network.actor_network.parameters():
        param.requires_grad_(False)
    for param in world_network.f.parameters():
        param.requires_grad_(True)
    for param in world_network.mlp_reward.parameters():
        param.requires_grad_(True)

    optimizer = torch.optim.Adam(world_network.parameters(), lr=.0001, weight_decay=0)

    # Training loop
    for i in range(NUM_EPOCHS):
        epoch_loss = 0
        for batch_idx, (u_batch, z_batch) in enumerate(my_dataloader):
            optimizer.zero_grad()
            x_batch = world_network.forward(u_batch)
            z_batch_centered = z_batch - torch.mean(z_batch, dim=1, keepdim=True)
            x_batch_centered = x_batch - torch.mean(x_batch, dim=1, keepdim=True)
            loss = torch.nn.MSELoss()(x_batch, z_batch)
            epoch_loss += loss.item() / NUM_EPOCHS
            loss.backward()
            torch.nn.utils.clip_grad_norm_(world_network.parameters(), max_norm=.1)
            optimizer.step()

        if i % 50 == 0:
            outputs = world_network.forward(inputs)
            print('Epoch: {}/{}.............'.format(i, NUM_EPOCHS), end=' ')
            labels_centered = labels - torch.mean(labels, dim=1, keepdim=True)
            print("mse_z: {:.8f}".format(
                torch.nn.MSELoss()(outputs, labels).item() / torch.nn.MSELoss()(torch.zeros_like(labels_centered),
                                                                                labels_centered).item()))
            torch.save(world_network, 'world_network.pth')
            torch.save(world_network.actor_network, 'actor_network.pth')

    # TRAIN ACTOR
    for param in world_network.actor_network.parameters():
        param.requires_grad_(True)
    for param in world_network.f.parameters():
        param.requires_grad_(False)
    for param in world_network.mlp_reward.parameters():
        param.requires_grad_(False)

    optimizer = torch.optim.Adam(world_network.parameters(), lr=0.0001)

    for i in range(NUM_EPOCHS):
        epoch_loss = 0
        for batch_idx, (u_batch, z_batch) in enumerate(my_dataloader):
            optimizer.zero_grad()
            x_batch = world_network.forward(u_batch)

            loss = -torch.sum(x_batch[:, :, -1])
            epoch_loss += loss.item() / NUM_EPOCHS
            loss.backward()
            torch.nn.utils.clip_grad_norm_(world_network.parameters(), max_norm=.1)
            optimizer.step()
        if i % 50 == 0:
            outputs = world_network.forward(inputs)
            print('Epoch: {}/{}.............'.format(i, NUM_EPOCHS), end=' ')

            print("Reward: {:.8f}".format(torch.mean(torch.sum(outputs[:, :, -1], dim=[1]))))
            torch.save(world_network, 'world_network.pth')
            torch.save(world_network.actor_network, 'actor_network.pth')

