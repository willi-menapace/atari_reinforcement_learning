#!/usr/bin/env python3
import gym
import ptan
import argparse
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

import torch.optim as optim

from tensorboardX import SummaryWriter

from lib import common

from lib.common import *
from lib.rainbow_model import RainbowDQN

def calc_loss(batch, batch_weights, net, tgt_net, gamma, device="cpu"):
    states, actions, rewards, dones, next_states = common.unpack_batch(batch)
    batch_size = len(batch)

    states_v = torch.tensor(states).to(device)
    actions_v = torch.tensor(actions).to(device)
    next_states_v = torch.tensor(next_states).to(device)
    #batch_weights_v = torch.tensor(batch_weights).to(device)

    # next state distribution
    # dueling arch -- actions from main net, distr from tgt_net

    # calc at once both next and cur states
    distr_v, qvals_v = net.both(torch.cat((states_v, next_states_v)))
    next_qvals_v = qvals_v[batch_size:]
    distr_v = distr_v[:batch_size]

    next_actions_v = next_qvals_v.max(1)[1]
    next_distr_v = tgt_net(next_states_v)
    next_best_distr_v = next_distr_v[range(batch_size), next_actions_v.data]
    next_best_distr_v = tgt_net.apply_softmax(next_best_distr_v)
    next_best_distr = next_best_distr_v.data.cpu().numpy()

    dones = dones.astype(np.bool)

    # project our distribution using Bellman update
    proj_distr = common.distr_projection(next_best_distr, rewards, dones, Vmin, Vmax, N_ATOMS, gamma)

    # calculate net output
    state_action_values = distr_v[range(batch_size), actions_v.data]
    state_log_sm_v = F.log_softmax(state_action_values, dim=1)
    proj_distr_v = torch.tensor(proj_distr).to(device)

    loss_v = -state_log_sm_v * proj_distr_v
    #loss_v = batch_weights_v * loss_v.sum(dim=1)
    return loss_v.mean(), loss_v + 1e-5


if __name__ == "__main__":
    params = common.HYPERPARAMS['pacman']

    parser = argparse.ArgumentParser()
    parser.add_argument("--cuda", default=False, action="store_true", help="Enable cuda")
    args = parser.parse_args()
    device = torch.device("cuda" if args.cuda else "cpu")

    wrapper = params["env_wrapper_train"]

    env = gym.make(params['env_name'])
    env = wrapper(env)

    writer = SummaryWriter(comment="-" + params['run_name'] + "-rainbow")
    net = RainbowDQN(env.observation_space.shape, env.action_space.n).to(device)

    agent = ptan.agent.DQNAgent(lambda x: net.qvals(x), ptan.actions.ArgmaxActionSelector(), device=device)

    exp_source = ptan.experience.ExperienceSourceFirstLast(env, agent, gamma=params['gamma'], steps_count=REWARD_STEPS)
    #buffer = ptan.experience.PrioritizedReplayBuffer(exp_source, params['replay_size'], PRIO_REPLAY_ALPHA)
    buffer = ptan.experience.ExperienceReplayBuffer(exp_source, buffer_size=params['replay_size'])
    optimizer = optim.Adam(net.parameters(), lr=params['learning_rate'])

    frame_idx = 0
    beta = BETA_START

    #If we are requested to continue training from an old checkpoint, load it
    saves_filename = params['resume_from']
    if saves_filename is not None:
        frame_idx = int(saves_filename.split("_")[1].split(".")[0])

        print("Loading network and optimizer {}".format(saves_filename))
        net.load_state_dict(torch.load(params["save_dir"] + saves_filename))
        #tgt_net.sync()
        optimizer.load_state_dict(torch.load(params["save_dir"] + saves_filename + ".optimizer"))

        for param_group in optimizer.param_groups:
            print("Learning rate correctly updated!")
            param_group['lr'] = params['learning_rate']

    tgt_net = ptan.agent.TargetNet(net)

    with common.RewardTracker(writer, params['stop_reward']) as reward_tracker:
        while True:
            frame_idx += 1
            buffer.populate(1)
            beta = min(1.0, BETA_START + frame_idx * (1.0 - BETA_START) / BETA_FRAMES)

            new_rewards = exp_source.pop_total_rewards()
            if new_rewards:
                if reward_tracker.reward(new_rewards[0], frame_idx):
                    break

            if len(buffer) <= params['replay_initial']:
                continue

            optimizer.zero_grad()
            #batch, batch_indices, batch_weights = buffer.sample(params['batch_size'], beta)
            batch = buffer.sample(params['batch_size'])

            if frame_idx % params['qvalues_estimation_interval'] == 0:
                avg_qvalues = calc_avg_qval(batch, net, device=device)
                writer.add_scalar("Batch qvalues", avg_qvalues, frame_idx)

            #loss_v, sample_prios_v = calc_loss(batch, batch_weights, net, tgt_net.target_model,
            #                                   params['gamma'] ** REWARD_STEPS, device=device)
            loss_v, sample_prios_v = calc_loss(batch, 0, net, tgt_net.target_model,
                                               params['gamma'] ** REWARD_STEPS, device=device)

            loss_v.backward()
            optimizer.step()
            #buffer.update_priorities(batch_indices, sample_prios_v.data.cpu().numpy())

            if frame_idx % params['target_net_sync'] == 0:
                tgt_net.sync()

            if frame_idx % params['save_interval'] == 0:
                common.save_net(net, optimizer, params['save_dir'], "{}_{}.dat".format(params['run_name'], frame_idx))

    common.save_net(net, optimizer, params['save_dir'], "best.dat")