import torch
from torch.utils.data import TensorDataset, DataLoader
import gymnasium as gym
import numpy as np
import torch.nn as nn
from Models.actor import *
from Models.world import *
# %%

device = 'cuda'

MEDIUM_MAZE = [
    [1, 1, 1, 1, 1, 1, 1, 1],
    [1, 0, 0, 0, 0, 0, 0, 1],
    [1, 0, 0, 0, 0, 0, 0, 1],
    [1, 0, 0, 0, 0, 0, 0, 1],
    [1, 0, 0, 0, 0, 0, 0, 1],
    [1, 0, 0, 0, 0, 0, 0, 1],
    [1, 0, 0, 0, 0, 0, 0, 1],
    [1, 1, 1, 1, 1, 1, 1, 1]]

# Initialize environment with this map for training.
env = gym.make('PointMaze_MediumDense-v3',
               max_episode_steps=100, maze_map=MEDIUM_MAZE,
               render_mode='rgb_array'
               )

# Initialize networks
world_network = WorldNet(actor_network = ActorNet().to(device),device = device).to(device)

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
            new_state, reward, _, _, _ = env.step(action.detach().cpu().numpy())
            new_state = torch.tensor(np.concatenate([new_state['observation'], new_state['desired_goal']])).float()
    
            # Add data 

            labels[episode, step, :] = torch.hstack([state, torch.tensor(reward).float()])
    
            # Set current state to new state
            state = new_state.clone()
            
    inputs = inputs.to(device)
    labels = labels.to(device)
    return inputs, labels


# PRETRAIN ACTOR
NUM_EPOCHS = 10000

for run in range(1):
    # Train world model
    print('Training world net')
    inputs, labels = generate_experience( world_network.actor_network,device=device)
    my_dataset = TensorDataset(inputs, labels)  #
    my_dataloader = DataLoader(my_dataset, batch_size=256, shuffle=True)
    
    # Turn gradients for actor
    for param in world_network.actor_network.parameters():
        param.requires_grad_(False)
    for param in world_network.f.parameters():
        param.requires_grad_(True)
    for param in world_network.mlp_reward.parameters():
        param.requires_grad_(True)

    optimizer = torch.optim.Adam(world_network.parameters(), lr=.001, weight_decay=0)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3000, gamma=0.1)
    # Training loop
    for i in range(NUM_EPOCHS):
        epoch_loss = 0
        for batch_idx, (u_batch, z_batch) in enumerate(my_dataloader):
            optimizer.zero_grad()
            x_batch = world_network.forward(u_batch)
            z_batch_centered  = z_batch - torch.mean(z_batch,dim=1,keepdim=True)
            x_batch_centered = x_batch - torch.mean(x_batch,dim=1,keepdim=True)
            loss = torch.nn.MSELoss()(x_batch, z_batch) 
            epoch_loss += loss.item() / NUM_EPOCHS
            loss.backward()
            torch.nn.utils.clip_grad_norm_(world_network.parameters(), max_norm=.1)
            optimizer.step()
    
        if i % 50 == 0:
            outputs = world_network.forward(inputs)
            print('Epoch: {}/{}.............'.format(i, NUM_EPOCHS), end=' ')
            labels_centered  = labels - torch.mean(labels,dim=1,keepdim=True)
            print("mse_z: {:.8f}".format(torch.nn.MSELoss()(outputs,labels).item()/ torch.nn.MSELoss()(torch.zeros_like(labels_centered), labels_centered).item()))
            torch.save(world_network,'world_network.pth')
            torch.save(world_network.actor_network, 'actor_network.pth')
        scheduler.step()

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
            #loss = -torch.nn.MSELoss()(x_batch, z_batch) 

            epoch_loss += loss.item() / NUM_EPOCHS
            loss.backward()
            torch.nn.utils.clip_grad_norm_(world_network.parameters(), max_norm=.1)
            optimizer.step()
        if i % 50 == 0:
            outputs = world_network.forward(inputs)
            print('Epoch: {}/{}.............'.format(i, NUM_EPOCHS), end=' ')
            labels_centered  = labels - torch.mean(labels,dim=1,keepdim=True)

            #print("mse_z: {:.8f}".format(-torch.nn.MSELoss()(outputs,labels).item()/ torch.nn.MSELoss()(torch.zeros_like(labels_centered), labels_centered).item()))

            print("Reward: {:.8f}".format(torch.mean(torch.sum(outputs[:, :, -1],dim=[1]))))
            torch.save(world_network, 'world_network.pth')
            torch.save(world_network.actor_network, 'actor_network.pth')

