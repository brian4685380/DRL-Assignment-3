from nes_py.wrappers import JoypadSpace
import gym_super_mario_bros
from gym_super_mario_bros.actions import COMPLEX_MOVEMENT
import numpy as np
from PIL import Image
import torch
from QNetwork import QNetwork
from constant import *
from utils import *
from PrioritizedBuffer import PrioritizedBuffer
from torch.optim import Adam
from tqdm import trange
import time
from wrappers import wrap_environment

def render_rgb(env):
    return Image.fromarray(env.render(mode='rgb_array'))

env = wrap_environment("SuperMarioBros-v0", COMPLEX_MOVEMENT)
env.reset()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
input_shape = env.observation_space.shape
qnet = QNetwork(input_shape, env.action_space.n).to(device)
target_qnet = QNetwork(input_shape, env.action_space.n).to(device)
qnet_dir = "mario_dqn_model.pth"
target_qnet_dir = "mario_dqn_target_model.pth"

replay_buffer = PrioritizedBuffer(capacity=MEMORY_CAPACITY)
optimizer = Adam(qnet.parameters(), lr=LEARNING_RATE)

reward_history = []
best_average_reward = -np.inf

def update(steps, beta):
    if len(replay_buffer) > INITIAL_LEARNING:
        if steps % TARGET_UPDATE_FREQUENCY == 0:
            target_qnet.load_state_dict(qnet.state_dict())
        optimizer.zero_grad()
        batch = replay_buffer.sample(BATCH_SIZE, beta)
        states, actions, rewards, next_states, dones, idx, weights = batch
        states = states.clone().detach().float().to(device)
        actions = actions.clone().detach().long().to(device)
        rewards = rewards.clone().detach().float().to(device)
        next_states = next_states.clone().detach().float().to(device)
        dones = dones.clone().detach().float().to(device)
        weights = weights.clone().detach().float().to(device)
        q_values = qnet(states)
        next_q_values = target_qnet(next_states)
        q_values = q_values.gather(1, actions)
        next_q_values = next_q_values.max(dim=1)[0]
        expected_q_values = rewards + GAMMA * next_q_values * (1 - dones)
        loss = (q_values - expected_q_values.detach()).pow(2) * weights
        priorities = (loss + 1e-5).view(-1)
        loss = loss.mean()
        loss.backward()
        replay_buffer.update_priorities(idx, priorities)
        optimizer.step()



if __name__ == "__main__":
    tic = time.time()
    for episode in trange(NUM_EPISODES):
        episode_reward = 0.0
        state = env.reset()
        state_np = state
        done = False
        steps = 0
        while not done:
            epsilon = update_epsilon(steps)
            if len(replay_buffer) > BATCH_SIZE:
                beta = update_beta(steps)
            else:
                beta = BETA_START
            
            if np.random.rand() < epsilon:
                action = env.action_space.sample()
            else:
                state_np = state 
                state = torch.from_numpy(state).unsqueeze(0).float().to(device, non_blocking=True)
                with torch.no_grad():
                    q_values = qnet(state)
                action = q_values.max(1)[1].item()
            next_state, reward, done, _ = env.step(action)
            replay_buffer.push(state_np, action, reward, next_state, done)
            state = next_state
            episode_reward += reward
            steps += 1
            # print(f"step: {steps} | action: {action} | reward: {reward} | epsilon: {epsilon:.4f} | beta: {beta:.4f}")
            update(steps, beta)
        reward_history.append(episode_reward)    
        toc = time.time()
        if episode % 100 == 0:
            average_reward = np.mean(reward_history[-100:])
            if average_reward > best_average_reward:
                best_average_reward = average_reward
                torch.save(qnet.state_dict(), qnet_dir)
                torch.save(target_qnet.state_dict(), target_qnet_dir)
        print(f"Episode {episode} | average reward: {average_reward:.4f} | reward: {episode_reward:.2f} | time: {toc - tic:.2f}s")
        tic = time.time()
    



        


