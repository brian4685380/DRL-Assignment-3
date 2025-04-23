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

def render_rgb(env):
    return Image.fromarray(env.render(mode='rgb_array'))

env = gym_super_mario_bros.make('SuperMarioBros-v0')
env = JoypadSpace(env, COMPLEX_MOVEMENT)
env.reset()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
input_shape = env.observation_space.shape  # (H, W, C)
input_shape = (input_shape[2], input_shape[0], input_shape[1])  # → (C, H, W)

qnet = QNetwork(input_shape, env.action_space.n).to(device)
target_qnet = QNetwork(input_shape, env.action_space.n).to(device)
qnet_dir = "mario_dqn_model.pth"
target_qnet_dir = "mario_dqn_target_model.pth"

replay_buffer = PrioritizedBuffer(capacity=MEMORY_CAPACITY)
optimizer = Adam(qnet.parameters(), lr=LEARNING_RATE)

def update(steps, beta):
    if len(replay_buffer) > INITIAL_LEARNING:
        if steps % TARGET_UPDATE_FREQUENCY == 0:
            target_qnet.load_state_dict(qnet.state_dict())
        optimizer.zero_grad()
        batch = replay_buffer.sample(BATCH_SIZE, beta)
        states, actions, rewards, next_states, dones, idx, weights = batch
        states = torch.tensor(states, dtype=torch.float32).to(device)
        actions = torch.tensor(actions, dtype=torch.int64).to(device)
        rewards = torch.tensor(rewards, dtype=torch.float32).to(device)
        next_states = torch.tensor(next_states, dtype=torch.float32).to(device)
        dones = torch.tensor(dones, dtype=torch.float32).to(device)
        weights = torch.tensor(weights, dtype=torch.float32).to(device)
        q_values = qnet(states)
        next_q_values = target_qnet(next_states)
        q_values = q_values.gather(1, actions)
        next_q_values = next_q_values.max(dim=1)[0]
        expected_q_values = rewards + GAMMA * next_q_values * (1 - dones)
        loss = (q_values - expected_q_values.detach()).pow(2) * weights
        priorities = loss + 1e-5
        loss = loss.mean()
        loss.backward()
        replay_buffer.update_priorities(idx, priorities)
        optimizer.step()

reward_history = []
best_reward = -np.inf
def run_episode():
    episode_reward = 0.0
    state = env.reset()
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
            state = torch.FloatTensor(state).to(device).unsqueeze(0)
            q_values = qnet(state)
            action = q_values.max(1)[1].item()
        next_state, reward, done, _ = env.step(action)
        replay_buffer.push(state, action, reward, next_state, done)
        state = next_state
        episode_reward += reward
        steps += 1
        update(steps, beta)
    reward_history.append(episode_reward)
    if episode_reward > best_reward:
        best_reward = episode_reward
        torch.save(qnet.state_dict(), qnet_dir)
        torch.save(target_qnet.state_dict(), target_qnet_dir)
        if episode % 100 == 0:
            average_reward = np.mean(reward_history[-100:])
            print(f"Episode {episode} | average reward: {average_reward:.4f} | best reward: {best_reward:.2f}")

if __name__ == "__main__":
    for episode in range(NUM_EPISODES):
        run_episode()
    



        


