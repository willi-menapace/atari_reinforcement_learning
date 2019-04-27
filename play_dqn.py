#!/usr/bin/env python3
import gym
import ptan
import argparse
import numpy as np
import time
import matplotlib.pyplot as plt

import torch
import torch.optim as optim

from tensorboardX import SummaryWriter

#todos
#le rewards vanno in clipping a 10, calcolare un nuovo limite massimo
#quando non ci sono robe da mangiare per lui andare in un fantasma o meno non cambia
#bisogna dare reward negativa alla morte

from lib import dqn_model, common, rainbow_model

dump_images = True
dump_directory = "screenshots/"

if __name__ == "__main__":

    saves_filename = "pacman_9700000.dat"
    step_count = int(saves_filename.split("_")[1].split(".")[0])

    params = common.HYPERPARAMS['pacman']
    parser = argparse.ArgumentParser()
    parser.add_argument("--cuda", default=False, action="store_true", help="Enable cuda")
    args = parser.parse_args()
    device = torch.device("cuda" if args.cuda else "cpu")

    wrapper = params["env_wrapper_test"]

    env = gym.make(params['env_name'])
    env = wrapper(env)

    #net = dqn_model.DQN(env.observation_space.shape, env.action_space.n).to(device)
    net = rainbow_model.RainbowDQN(env.observation_space.shape, env.action_space.n).to(device)

    frame_idx = 0

    #Loads saved net
    net.load_state_dict(torch.load(params["save_dir"] + saves_filename))

    obs = env.reset()
    done = False
    while True:
        if np.random.randint(0, 5000000000, 1) == 0:
            plt.imshow(np.asarray(obs)[0,:,:], cmap="gray")
            plt.savefig("{}{}_{}.png".format(dump_directory, params["run_name"], frame_idx))
            frame_idx += 1

        tensor = torch.tensor(np.expand_dims(np.asarray(obs), axis=0)).to(device)
        q_values = net.qvals(tensor).cpu().data.numpy()[0]
        #print(q_values)

        action = np.argmax(q_values)
        obs, reward, done, info = env.step(action)
        print(reward)

        time.sleep(1/30)

        if done:
            obs = env.reset()

        env.render()