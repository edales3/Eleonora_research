import numpy as np
import matplotlib.pyplot as plt
import math
import argparse
import copy
from collections import namedtuple

import img_utils
from gridworld import *
from maxent_irl_ent import *

Step = namedtuple('Step', 'cur_state action next_state reward done')

PARSER = argparse.ArgumentParser(description=None)
PARSER.add_argument('-hei', '--height', default=5, type=int, help='height of the gridworld')
PARSER.add_argument('-wid', '--width', default=5, type=int, help='width of the gridworld')
PARSER.add_argument('-g', '--gamma', default=0.8, type=float, help='discount factor')
PARSER.add_argument('-a', '--act_random', default=0.3, type=float, help='probability of acting randomly')
PARSER.add_argument('-t', '--n_trajs', default=100, type=int, help='number of expert trajectories')
PARSER.add_argument('-l', '--l_traj', default=20, type=int, help='length of expert trajectory')
PARSER.add_argument('--rand_start', dest='rand_start', action='store_true',
                    help='when sampling trajectories, randomly pick start positions')
PARSER.add_argument('--no-rand_start', dest='rand_start', action='store_false',
                    help='when sampling trajectories, fix start positions')
PARSER.set_defaults(rand_start=True)
PARSER.add_argument('-lr', '--learning_rate', default=0.01, type=float, help='learning rate')
PARSER.add_argument('-ni', '--n_iters', default=20, type=int, help='number of iterations')
ARGS = PARSER.parse_args()
print(ARGS)

GAMMA = ARGS.gamma
ACT_RAND = ARGS.act_random
R_MAX = 1  # the constant r_max does not affect much the recovered reward distribution
H = ARGS.height
W = ARGS.width
N_TRAJS = ARGS.n_trajs
L_TRAJ = ARGS.l_traj
RAND_START = ARGS.rand_start
LEARNING_RATE = ARGS.learning_rate
N_ITERS = ARGS.n_iters


def generate_demonstrations(gw, policy, n_trajs=100, len_traj=20, rand_start=False, start_pos=[0, 0]):
    """gather expert demonstrations

      :param gw: Gridworld - the environment
      :param policy: Nx1 matrix
      :param n_trajs: int - number of trajectories to generate
      :param len_traj: int - number of steps in each trajectory
      :param rand_start: bool - randomly picking start position or not
      :param start_pos: 2x1 list - set start position, default [0,0]
      :return: trajs: a list of trajectories - each element in the list is a list of Steps representing an episode
    """

    trajs = []
    for i in range(n_trajs):
        if rand_start:
            # override start_pos
            start_pos = [np.random.randint(0, gw.height), np.random.randint(0, gw.width)]

        episode = []
        gw.reset(start_pos)
        cur_state = start_pos
        cur_state, action, next_state, reward, is_done = gw.step(int(policy[gw.pos2idx(cur_state)]))
        episode.append(
            Step(cur_state=gw.pos2idx(cur_state), action=action, next_state=gw.pos2idx(next_state), reward=reward,
                 done=is_done))
        # while not is_done:
        for _ in range(len_traj):
            cur_state, action, next_state, reward, is_done = gw.step(int(policy[gw.pos2idx(cur_state)]))
            episode.append(
                Step(cur_state=gw.pos2idx(cur_state), action=action, next_state=gw.pos2idx(next_state), reward=reward,
                     done=is_done))
            if is_done:
                break
        trajs.append(episode)
    return trajs


def value_iteration(P_a, rewards, gamma, error=0.01, deterministic=True):
  """ Value iteration through dynamic programming

  :param P_a: NxNxN_ACTIONS transition probabilities matrix -
                              P_a[s0, s1, a] is the transition prob of
                              landing at state s1 when taking action
                              a at state s0
  :param rewards: Nx1 matrix - rewards for all the states
  :param gamma: float - RL discount
  :param error: float - threshold for a stop
  :param deterministic: bool - to return deterministic policy or stochastic policy
  :return: values    Nx1 matrix - estimated values
           policy    Nx1 (NxN_ACTIONS if non-det) matrix - policy
  """
  N_STATES, _, N_ACTIONS = np.shape(P_a)

  values = np.zeros([N_STATES])

  # estimate values
  while True:
    values_tmp = values.copy()

    for s in range(N_STATES):
      v_s = []
      values[s] = max([sum([P_a[s, s1, a]*(rewards[s1] + gamma*values_tmp[s1]) for s1 in range(N_STATES)]) for a in range(N_ACTIONS)])

    if max([abs(values[s] - values_tmp[s]) for s in range(N_STATES)]) < error:
      break


  if deterministic:
    # generate deterministic policy
    policy = np.zeros([N_STATES])
    for s in range(N_STATES):
      policy[s] = np.argmax([sum([P_a[s, s1, a]*(rewards[s1]+gamma*values[s1])
                                  for s1 in range(N_STATES)])
                                  for a in range(N_ACTIONS)])

    return values, policy.astype(int)
  else:
    # generate stochastic policy
    policy = np.zeros([N_STATES, N_ACTIONS])
    for s in range(N_STATES):
      v_s = np.array([sum([P_a[s, s1, a]*(rewards[s1] + gamma*values[s1]) for s1 in range(N_STATES)]) for a in range(N_ACTIONS)])
      policy[s,:] = np.transpose(v_s/np.sum(v_s))
    return values, policy


def GetValue(policy, transition_probabilities, reward, discount, threshold=1e-2, deterministic=True):
    """

    :param policy: List of actions ints for each state.
    :param n_states: Number of states. int.
    :param transition_probabilities: Function taking (s_prev, s_next, state) to transition probabilities.
    :param reward: Vector of rewards for each state.
    :param discount: MDP discount factor. float.
    :param threshold:Convergence threshold, default 1e-2. float.
    :return:Array of values for each state
    """
    N_STATES, _, N_ACTIONS = np.shape(transition_probabilities)

    v = np.zeros(N_STATES)

    if deterministic:
        diff = float("inf")
        while diff > threshold:
            diff = 0
            for s0 in range(N_STATES):
                vs = copy.deepcopy(v[s0])
                a = policy[s0]
                v[s0] = sum(transition_probabilities[s0, s1, a] * (reward[s1] + discount * v[s1]) for s1 in range(N_STATES))
                diff = max(diff, abs(vs - v[s0]))
    else:
        diff = float("inf")
        while diff > threshold:
            diff = 0
            for s0 in range(N_STATES):
                vs = copy.deepcopy(v[s0])
                v[s0] = sum([sum([transition_probabilities[s0, s1, a] * (reward[s1] + discount * v[s1]) * policy[s0,a] for a in range(N_ACTIONS)]) for s1 in range(N_STATES)])
                diff = max(diff, abs(vs - v[s0]))

    return v

def main():
    N_STATES = H * W
    N_ACTIONS = 5

    # init the gridworld
    # rmap_gt is the ground truth for rewards

    #"""
    rmap_gt = np.zeros([H, W])
    rmap_gt[H - 1, W - 1] = R_MAX
    rmap_gt[H - 1, 0] = R_MAX

    ACT_RAND = 0

    gw = gridworld.GridWorld(rmap_gt, {}, 1 - ACT_RAND)

    rewards_gt = np.reshape(rmap_gt, H * W, order='F')
    """
    rmap_gt = np.zeros([H, W])
    rmap_gt[H - 2, W - 2] = R_MAX
    rmap_gt[1, 1] = R_MAX
    gw = gridworld.GridWorld(rmap_gt, {}, 1 - ACT_RAND)
    rewards_gt = np.reshape(rmap_gt, H * W, order='F')
    P_a = gw.get_transition_mat()
    values_gt, policy_gt = value_iteration(P_a, rewards_gt, GAMMA, error=0.01, deterministic=True)
    rewards_gt = normalize(values_gt)
    gw = gridworld.GridWorld(np.reshape(rewards_gt, (H, W), order='F'), {}, 1 - ACT_RAND)
    """ #
    P_a = gw.get_transition_mat()

    values_gt, policy_gt = value_iteration(P_a, rewards_gt, GAMMA, error=0.01, deterministic=True)

    # use identity matrix as feature
    feat_map = np.eye(N_STATES)

    np.random.seed(1)

    eg = []
    tg = []
    unseen = []
    for i in range(59,60):
        print("i = {}".format(i+1))
        N_TRAJS = 100
        L_TRAJ = (i+1)
        trajs = generate_demonstrations(gw, policy_gt, n_trajs=N_TRAJS, len_traj=L_TRAJ, rand_start=RAND_START)
        #rewards = maxent_irl(feat_map, P_a, GAMMA, trajs, LEARNING_RATE, N_ITERS)
        rewards_ent, policy_ent, n_unseen = maxent_irl_ent(feat_map, P_a, GAMMA, trajs, LEARNING_RATE, N_ITERS, deterministic=True)
        value_ent = GetValue(policy_ent, P_a, rewards_gt, GAMMA,deterministic=True)
        _, policy_theta = value_iteration(P_a, rewards_ent, GAMMA, error=0.01, deterministic=True)
        value_theta = GetValue(policy_theta, P_a, rewards_gt, GAMMA, deterministic=True)
        eg.append(np.linalg.norm(value_ent - values_gt) / np.linalg.norm(values_gt))
        tg.append(np.linalg.norm(value_theta - values_gt) / np.linalg.norm(values_gt))
        unseen.append(n_unseen)

    unseen = np.array(unseen)
    plt.figure(1)
    plt.plot(eg, marker='.')
    plt.plot(tg, marker='.')
    #plt.plot((unseen / max(unseen)), marker='.')
    plt.grid(True)
    plt.ylabel('||Vgt - V||2/||Vgt||2')
    plt.xlabel('length_expert_demos')
    plt.legend(['V = Vent', 'V = Vtheta'])#, '#unseen'])
    plt.show()

    plt.figure(2)
    plt.plot(values_gt, marker='.')
    plt.plot(value_theta, marker='.')
    plt.plot(value_ent, marker='.')
    plt.grid(True)
    plt.ylabel('V')
    plt.xlabel('s')
    plt.legend(['Vgt', 'Vtheta', 'Vent'])
    plt.show()

    # plots
    plt.figure(figsize=(25, 5))
    plt.subplot(1, 5, 1)
    img_utils.heatmap2d(np.reshape(rewards_gt, (H, W), order='F'), 'Rewards Map - Ground Truth', block=False)
    plt.subplot(1, 5, 2)
    img_utils.heatmap2d(np.reshape(rewards_ent, (H, W), order='F'), 'Reward_Ent - Recovered', block=False)
    plt.subplot(1, 5, 3)
    img_utils.heatmap2d(np.reshape(values_gt, (H, W), order='F'), 'Value - Ground Truth', block=False)
    plt.subplot(1, 5, 4)
    img_utils.heatmap2d(np.reshape(value_ent, (H, W), order='F'), 'Value Ent - Recovered', block=False)
    plt.subplot(1, 5, 5)
    img_utils.heatmap2d(np.reshape(value_theta, (H, W), order='F'), 'Value Theta - Recovered', block=False)
    plt.show()
    #plt.subplot(2, 2, 4)
    #img_utils.heatmap3d(np.reshape(rewards_ent, (H,W), order='F'), 'Reward Map - Recovered')


if __name__ == "__main__":
    main()
