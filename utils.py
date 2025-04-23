from constant import *
import numpy as np
def update_epsilon(episode):
    episode_start = EPSILON_START
    episode_final = EPSILON_FINAL
    epsilon_decay = EPSILON_DECAY
    return episode_final + (episode_start - episode_final) * np.exp(-1. * (episode + 1) / epsilon_decay)

def update_beta(episode):
    beta_start = BETA_START
    beta_frames = BETA_FRAMES
    beta_final = 1.0
    beta = beta_start + (beta_final - beta_start) * episode / beta_frames
    return min(beta, beta_final)