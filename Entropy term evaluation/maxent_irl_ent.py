import numpy as np
import gridworld
import img_utils
import copy
from utils import *



def maxentMDP(n_states, reward, n_actions, gamma, P_a, deterministic=True):
    """solves the entropy-regularized MDP

    :param n_states: State space dimension. int
    :param reward: Reward grid HxW. Nx1 vector
    :param n_actions: Action space dimension. int
    :param gamma: float - RL discount factor
    :param P_a: NxNxN_ACTIONS matrix - P_a[s0, s1, a] is the transition prob of
                                       landing at state s1 when taking action
                                       a at state s0
    :param deterministic: Boolean value for deterministic or stochastic policy
    :return: policy solution of the entropy-regularized MDP
    """
    V = np.array([-1e50]*n_states)

    diff = np.ones((n_states,))
    while (diff > 1e-4).all():  # Iterate until convergence.
        new_V = copy.deepcopy(reward)
        for s0 in range(n_states):
            for a in range(n_actions):
                new_V[s0] = softmax(new_V[s0], (sum([P_a[s0, s1, a] * (reward[s1] + V[s1]) for s1 in range(n_states)])))

        # # This seems to diverge, so we z-score it
        new_V = (new_V - new_V.mean()) / new_V.std()

        diff = abs(V - new_V)
        V = copy.deepcopy(new_V)

    # Extract Q_soft from V_soft using equation 9.2 of Zeibart thesis
    Q = np.zeros((n_states, n_actions))
    for s0 in range(n_states):
        for a in range(n_actions):
            Q[s0, a] = sum([P_a[s0, s1, a] * (reward[s1] + V[s1]) for s1 in range(n_states)])

    # Extract policy using equation 9.1 of Zeibart thesis
    Q -= Q.max(axis=1).reshape((n_states, 1))  # For numerical stability.
    policy = np.exp(Q) / np.exp(Q).sum(axis=1).reshape((n_states, 1))
    if deterministic:
        return np.argmax(policy, axis=1)
    return policy


def softmax(x1, x2):
    """Soft-maximum calculation, from algorithm 9.2 in Ziebart's thesis

    :param x1: float
    :param x2: float
    :return: softmax(x1, x2)
    """

    max_x = max(x1, x2)
    min_x = min(x1, x2)
    return max_x + np.log(1 + np.exp(min_x - max_x))


def compute_state_visitation_freq1(P_a, trajs, policy, deterministic=True):
    """This version implements Algorithm 9.3 in Ziebart '10 thesis

    inputs:
    P_a     NxNxN_ACTIONS matrix - transition dynamics
    gamma   float - discount factor
    trajs   list of list of Steps - collected from expert
    policy  Nx1 vector (or NxN_ACTIONS if deterministic=False) - policy

    returns:
    p       Nx1 vector - state visitation frequencies
    """
    N_STATES, _, N_ACTIONS = np.shape(P_a)

    D = np.zeros([N_STATES])

    P_0 = np.zeros_like(D)
    for traj in trajs:
        P_0[traj[0].cur_state] += 1
    P_0 = P_0 / len(trajs)

    diff = np.ones((N_STATES,))
    while (diff > 1e-4).all():
        D_new = copy.deepcopy(P_0)

        for s0 in range(N_STATES):
            if deterministic:
                for s1 in range(N_STATES):
                    D_new[s1] += D[s0] * P_a[s0, s1, policy[s0]]
            else:
                for a in range(N_ACTIONS):
                    for s1 in range(N_STATES):
                        D_new[s1] += D[s0] * policy[s0,a] * P_a[s0, s1, a]
        #D_new = (D_new - D_new.mean()) / D_new.std()

        diff = abs(D - D_new)
        D = copy.deepcopy(D_new)

    return D

def compute_state_visitation_freq2(P_a, trajs, policy, deterministic=True):
    """This version implements steps 4., 5. and 6. of Algorithm 1 in Ziebart '08 paper

    :param P_a: NxNxN_ACTIONS matrix - transition dynamics
    :param gamma: float - discount factor
    :param trajs: list of list of Steps - collected from expert
    :param policy: Nx1 vector (or NxN_ACTIONS if deterministic=False) - policy
    :param deterministic: Boolean - deterministic or stochastic policy
    :return: Nx1 vector - state visitation frequencies
    """

    N_STATES, _, N_ACTIONS = np.shape(P_a)

    T = len(trajs[0])
    # mu[s, t] is the prob of visiting state s at time t
    mu = np.zeros([N_STATES, T])

    for traj in trajs:
        mu[traj[0].cur_state, 0] += 1
    mu[:, 0] = mu[:, 0] / len(trajs)

    for s in range(N_STATES):
        for t in range(T - 1):
            if deterministic:
                mu[s, t + 1] = sum([mu[pre_s, t] * P_a[pre_s, s, int(policy[pre_s])] for pre_s in range(N_STATES)])
            else:
                mu[s, t + 1] = sum(
                    [sum([mu[pre_s, t] * P_a[pre_s, s, a1] * policy[pre_s, a1] for a1 in range(N_ACTIONS)]) for pre_s in
                     range(N_STATES)])
    D = np.sum(mu, 1)
    return D


def maxent_irl_ent(feat_map, P_a, gamma, trajs, lr, n_iters, deterministic=True):
    """Maximum Entropy Inverse Reinforcement Learning (Maxent IRL)

    :param feat_map: NxD matrix - the features for each state
    :param P_a: NxNxN_ACTIONS matrix - P_a[s0, s1, a] is the transition prob of
                                       landing at state s1 when taking action
                                       a at state s0
    :param gamma: float - RL discount factor
    :param trajs: expert demonstrations
    :param lr: float - learning rate
    :param n_iters: int - number of optimization epochs
    :param deterministic: Boolean - If the policy is deterministic or not
    :return: rewards: Nx1 vector - recoverred state rewards
    """
    N_STATES, _, N_ACTIONS = np.shape(P_a)

    # init parameters
    theta = np.random.uniform(size=(feat_map.shape[1],))

    # calc feature expectations
    feat_exp = np.zeros([feat_map.shape[1]])
    for episode in trajs:
        for step in episode:
            feat_exp += feat_map[step.cur_state, :]
    feat_exp = feat_exp / len(trajs)


    n_nonzero = np.count_nonzero(feat_exp)

    # training
    for iteration in range(n_iters):

        if iteration % (n_iters / 20) == 0:
            print('iteration: {}/{}'.format(iteration+1, n_iters))

        # compute reward function
        rewards = np.dot(feat_map, theta)

        # compute policy
        policy = maxentMDP(N_STATES, rewards, N_ACTIONS, gamma, P_a, deterministic=deterministic)

        # compute state visitation frequencies
        #svf = compute_state_visitation_freq1(P_a, trajs, policy, deterministic=deterministic)
        svf = compute_state_visitation_freq2(P_a, trajs, policy, deterministic=deterministic)

        # compute gradients
        grad = feat_exp - feat_map.T.dot(svf)

        # update params
        theta += lr * grad

    rewards = np.dot(feat_map, theta)

    return normalize(rewards), policy, N_STATES-n_nonzero
