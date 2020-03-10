"""Tabular AIRL algorithm validation"""
import numpy as np
import copy
from q_iteration import q_iteration, logsumexp, get_policy
from maxent_irl import tabular_maxent_irl, compute_visitation
from simple_env import random_env

def direct_method(env,ent_wt,true_sa_visits,discount,dim_obs,reward,transition,state_only):
# this version of the direct method solves a system of linear equations (as in slide 12)
    learned_rew, learned_q = tabular_maxent_irl(env, true_sa_visits, lr=0.01, num_itrs=1000,
                                                ent_wt=ent_wt, state_only=state_only,
                                                discount=discount)
    learned_policy = get_policy(learned_q, ent_wt=ent_wt)
    policy_irl = np.argmax(learned_policy, axis=1)
    reward_array_irl = np.array([reward[k, policy_irl[k]] for k in range(dim_obs)])
    transition_matrix = np.array([transition[j, policy_irl[j]] for j in range(dim_obs)])

    A = np.identity(dim_obs) - discount*transition_matrix
    b = reward_array_irl
    V_irl = np.linalg.solve(A,b)

    return V_irl


def value_iteration(dim_obs,reward,discount,transition):
# the value iteration algorithm does not solve a linear system but iterates on the Bellman equation

    eps = 10e-9
    V_star = np.zeros(dim_obs)
    delta = 2 * eps
    while delta > eps:
        delta_max = -1
        for i in range(dim_obs):
            v_prev = copy.deepcopy(V_star[i])
            allvalues = []
            for a in range(dim_act):
                allvalues.append(reward[i, a] + discount * np.dot(transition[i, a], V_star))
            V_star[i] = np.max(allvalues)
            if abs(V_star[i] - v_prev) > delta_max:
                delta_max = abs(V_star[i] - v_prev)
        delta = delta_max
    return V_star

if __name__ == "__main__":

    # Creating the environment using "simple_env.py", collecting true reward and transition probabilities
    env = random_env(10, 5, seed=1, terminate=False, t_sparsity=0.8)
    ent_wt = 0.1
    discount = 0.9
    dim_obs = env.observation_space.flat_dim
    dim_act = env.action_space.flat_dim
    reward = env.rew_matrix
    transition = env.transition_matrix

    # in the tabular environment the true state-action visitation frequency works as expert demonstration
    true_q,true_v = q_iteration(env, K=150, ent_wt=ent_wt, gamma=discount)
    true_sa_visits = compute_visitation(env, true_q, ent_wt=ent_wt, T=5, discount=discount)

    # Computing V_irl (state-only depencency) and V_irl_sa (state-action dependency) using the direct method
    # and V_star using value iteration
    V_irl = direct_method(env,ent_wt,true_sa_visits,discount,dim_obs,reward,transition,state_only=True)
    V_irl_sa = direct_method(env,ent_wt,true_sa_visits,discount,dim_obs,reward,transition,state_only=False)
    V_star = value_iteration(dim_obs,reward,discount,transition)
