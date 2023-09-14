### MDP Value Iteration and Policy Iteration
### Reference: https://web.stanford.edu/class/cs234/assignment1/index.html 
import numpy as np
import collections
collections.Callable = collections.abc.Callable
np.set_printoptions(precision=3)

"""
For policy_evaluation, policy_improvement, policy_iteration and value_iteration,
the parameters P, nS, nA, gamma are defined as follows:

	P: nested dictionary
		From gym.core.Environment
		For each pair of states in [1, nS] and actions in [1, nA], P[state][action] is a
		tuple of the form (probability, nextstate, reward, terminal) where
			- probability: float
				the probability of transitioning from "state" to "nextstate" with "action"
			- nextstate: int
				denotes the state we transition to (in range [0, nS - 1])
			- reward: int
				either 0 or 1, the reward for transitioning from "state" to
				"nextstate" with "action"
			- terminal: bool
			  True when "nextstate" is a terminal state (hole or goal), False otherwise
	nS: int
		number of states in the environment
	nA: int
		number of actions in the environment
	gamma: float
		Discount factor. Number in range [0, 1)
"""

def policy_evaluation(P, nS, nA, policy, gamma=0.9, tol=1e-8):
    value_function = np.zeros(nS)
    while True:
        diff = 0
        for state in range(nS):
            v = value_function[state]
            reward_actions = 0
            for action in range(nA):
                prob_action = policy[state][action]
                transitions = P[state][action]
                r_t = 0
                for t in transitions:
                    s_prime_prob = t[0]
                    s_prime = t[1]
                    s_prime_reward = t[2]
                    r_t += s_prime_prob * (s_prime_reward + gamma * value_function[s_prime])
                reward_actions += prob_action * r_t
            value_function[state] = reward_actions
            accuracy = abs(v - reward_actions)
            diff = max(diff, accuracy)
        if diff < tol:
            break

    return value_function

def policy_improvement(P, nS, nA, value_from_policy, gamma=0.9):
    """Given the value function from policy improve the policy.

    Parameters:
    -----------
    P, nS, nA, gamma:
        defined at beginning of file
    value_from_policy: np.ndarray
        The value calculated from the policy
    Returns:
    --------
    new_policy: np.ndarray[nS,nA]
        A 2D array of floats. Each float is the probability of the action
        to take in that state according to the environment dynamics and the 
        given value function.
    """

    new_policy = np.zeros([nS, nA])

    for state in range(nS):
        opt_r, best_action = 0, 0
        for action in range(nA):
            transitions = P[state][action]
            r_t = 0
            for t in transitions:
                s_prime_prob = t[0]
                s_prime = t[1]
                s_prime_reward = t[2]
                r_t += s_prime_prob * (s_prime_reward + gamma * value_from_policy[s_prime])
            if r_t > opt_r:
                opt_r = r_t
                best_action = action
        new_policy[state][best_action] = 1

    return new_policy


def policy_iteration(P, nS, nA, policy, gamma=0.9, tol=1e-8):
    """Runs policy iteration.

    You should call the policy_evaluation() and policy_improvement() methods to
    implement this method.

    Parameters
    ----------
    P, nS, nA, gamma:
        defined at beginning of file
    policy: policy to be updated
    tol: float
        tol parameter used in policy_evaluation()
    Returns:
    ----------
    new_policy: np.ndarray[nS,nA]
    V: np.ndarray[nS]
    """
    new_policy = policy.copy()
    V = np.zeros(nS)
    i = 0
    policy_norm = np.linalg.norm(new_policy - policy)
    while i == 0 or policy_norm > 0:
        V = policy_evaluation(P, nS, nA, policy, gamma, tol)
        new_policy = policy_improvement(P, nS, nA, V, gamma)
        policy_norm = np.linalg.norm(new_policy - policy)
        i += 1
        policy = new_policy  # Use updated policy for next iteration
    return new_policy, V

def value_iteration(P, nS, nA, V, gamma=0.9, tol=1e-8):
    """
    Learn value function and policy by using value iteration method for a given
    gamma and environment.

    Parameters:
    ----------
    P, nS, nA, gamma:
        defined at beginning of file
    V: value to be updated
    tol: float
        Terminate value iteration when
            max |value_function(s) - prev_value_function(s)| < tol
    Returns:
    ----------
    policy_new: np.ndarray[nS,nA]
    V_new: np.ndarray[nS]
    """
    
    V_new = V.copy()
    while True:
        diff = 0
        policy_new = np.zeros([nS, nA])
        for state in range(nS):
            v = V_new[state]
            max_reward, best_action = 0, 0
            for action in range(nA):
                transitions = P[state][action]
                r_t = 0
                for t in transitions:
                    s_prime_prob = t[0]
                    s_prime = t[1]
                    s_prime_reward = t[2]             
                    r_t += s_prime_prob * (s_prime_reward + gamma * V_new[s_prime])
                if r_t > max_reward:
                    max_reward = r_t
                    best_action = action
            V_new[state] = max_reward
            policy_new[state][best_action] = 1
            accuracy = abs(v - max_reward)
            diff = max(diff, accuracy)
        if diff < tol:
            break
    return policy_new, V_new

def render_single(env, policy, render = False, n_episodes=100):
    """
    Given a game envrionemnt of gym package, play multiple episodes of the game.
    An episode is over when the returned value for "done" = True.
    At each step, pick an action and collect the reward and new state from the game.

    Parameters:
    ----------
    env: gym.core.Environment
      Environment to play on. Must have nS, nA, and P as attributes.
    policy: np.array of shape [nS, nA]
      The action to take at a given state
    render: whether or not to render the game(it's slower to render the game)
    n_episodes: the number of episodes to play in the game. 
    Returns:
    ------
    total_rewards: the total number of rewards achieved in the game.
    -----
    Transition can be done using the function env.step(a) below with FIVE output parameters:
    ob, r, done, info, prob = env.step(a) 
    """
    total_rewards = 0
    for _ in range(n_episodes):
        ob = env.reset() 
        done = False
        while not done:
            if render:
                env.render()  
            action_index = 0
            for index, action in enumerate(policy[ob]):
                if action == 1:
                    action_index = index
                    break
            ob, reward, done, info, prob = env.step(action_index)
            total_rewards += reward
    return total_rewards