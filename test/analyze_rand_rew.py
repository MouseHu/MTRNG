import numpy as np
import pickle as pkl


file_dir = "../rewards.pkl"

with open(file_dir,"rb") as f:
    true_reward, random_reward = pkl.load(f)

print(true_reward.shape,random_reward.shape)