from nes_py.wrappers import JoypadSpace
import gym_super_mario_bros
from gym_super_mario_bros.actions import COMPLEX_MOVEMENT
import numpy as np
from PIL import Image
import torch
from DuelingQNetwork import DuelingQNetwork
from constant import *
from utils import *
from PrioritizedBuffer import PrioritizedBuffer
from torch.optim import Adam
import time
from wrappers import wrap_environment


def render_rgb(env):
    return Image.fromarray(env.render(mode='rgb_array'))

env = wrap_environment(
    "SuperMarioBros-v0",
     COMPLEX_MOVEMENT,
     skip = 4,
     num_stack=4
)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")
input_shape = env.observation_space.shape
qnet = DuelingQNetwork(input_shape, env.action_space.n).to(device)
target_qnet = DuelingQNetwork (input_shape, env.action_space.n).to(device)
qnet_dir = "mario_PER_dueling_ddqn_model.pth"
target_qnet_dir = "mario_PER_dueling_ddqn_target_model.pth"

replay_buffer = PrioritizedBuffer(capacity=MEMORY_CAPACITY)
optimizer = Adam(qnet.parameters(), lr=LEARNING_RATE)
qnet.train()
target_qnet.eval()

reward_history = []
best_reward = -np.inf

def update(global_steps):
    if len(replay_buffer) > INITIAL_LEARNING:
        if steps % TARGET_UPDATE_FREQUENCY == 0:
            target_qnet.load_state_dict(qnet.state_dict())
        optimizer.zero_grad()
        beta = update_beta(global_steps)
        states, actions, rewards, next_states, dones, weights, indices = replay_buffer.sample(BATCH_SIZE, beta)
        
        q_values = qnet(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        with torch.no_grad():
            next_action = qnet(next_states).argmax(dim=1, keepdim=True)
            next_q_values = target_qnet(next_states).gather(1, next_action).squeeze(1)
            target_q_values = rewards + (1 - dones) * GAMMA * next_q_values
            
        td_errors = q_values - target_q_values
        loss = ((td_errors * weights).pow(2) * weights).mean()
        loss.backward()
        optimizer.step()

        replay_buffer.update_priorities(indices, td_errors)    

if __name__ == "__main__":
    tic = time.time()
    global_steps = 0
    state = env.reset()
    for episode in range(NUM_EPISODES):
        env.reset()
        episode_reward = 0.0
        done = False
        steps = 0
        epsilon = update_epsilon(episode)
        while not done:
            if np.random.rand() < epsilon:
                action = env.action_space.sample()
            else:
                state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)
                with torch.no_grad():
                    q_values = qnet(state_tensor)
                action = q_values.max(dim=1)[1].item()
            next_state, reward, done, _ = env.step(action)
            replay_buffer.add(state, action, reward, next_state, done)
            state = next_state
            episode_reward += reward
            steps += 1
            global_steps += 1
            update(global_steps)
        reward_history.append(episode_reward)
        if episode_reward > best_reward:
            best_reward = episode_reward
            torch.save(qnet.state_dict(), qnet_dir)
            torch.save(target_qnet.state_dict(), target_qnet_dir)    
        average_reward = np.mean(reward_history[-100:])
        if episode % 10 == 0:
            toc = time.time()
            print(f"Episode {episode} | epsilon: {epsilon:.4f} | average reward: {average_reward:.4f} | reward: {episode_reward:.2f} | time: {toc - tic:.2f}s | best reward: {best_reward:.2f}")
        tic = time.time()
        pass