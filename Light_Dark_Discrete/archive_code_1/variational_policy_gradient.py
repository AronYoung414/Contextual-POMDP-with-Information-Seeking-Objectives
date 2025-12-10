import numpy as np
import time
import matplotlib.pyplot as plt
import pickle
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from scipy.sparse import csr_matrix

from random import choice
from random import seed
import os

from line_grid_environment import Environment


def set_random_seeds(sd=42):
    """Set random seeds for all libraries to ensure reproducibility"""
    np.random.seed(sd)
    torch.manual_seed(sd)
    torch.cuda.manual_seed(sd)
    torch.cuda.manual_seed_all(sd)
    # For Python's built-in random module
    seed(sd)


class PolicyNetwork(nn.Module):
    def __init__(self, obs_vocab_size, action_size, hidden_dim=64, max_seq_len=20):
        super(PolicyNetwork, self).__init__()
        self.obs_vocab_size = obs_vocab_size
        self.action_size = action_size
        self.hidden_dim = hidden_dim
        self.max_seq_len = max_seq_len

        # Embedding layer for observations
        self.obs_embedding = nn.Embedding(obs_vocab_size, hidden_dim)

        # LSTM to process observation sequences
        self.lstm = nn.LSTM(hidden_dim, hidden_dim, batch_first=True)

        # Output layer to produce action probabilities (only for non-end actions)
        self.action_head = nn.Linear(hidden_dim, self.action_size)

    def forward(self, obs_sequences, sequence_lengths=None):
        """
        Forward pass of the policy network
        Args:
            obs_sequences: Tensor of shape (batch_size, seq_len) containing observation indices
            sequence_lengths: Tensor of shape (batch_size,) containing actual sequence lengths
        Returns:
            action_probs: Tensor of shape (batch_size, action_size) containing action probabilities
        """
        # batch_size, seq_len = obs_sequences.shape

        # Embed observations
        embedded_obs = self.obs_embedding(obs_sequences)  # (batch_size, seq_len, hidden_dim)

        # Process through LSTM
        if sequence_lengths is not None:
            # Pack padded sequences for efficiency
            packed_embedded = nn.utils.rnn.pack_padded_sequence(
                embedded_obs, sequence_lengths, batch_first=True, enforce_sorted=False
            )

            lstm_out, (hidden, cell) = self.lstm(packed_embedded)
            # Use the last hidden state
            final_hidden = hidden[-1]  # (batch_size, hidden_dim)
        else:
            lstm_out, (hidden, cell) = self.lstm(embedded_obs)
            final_hidden = hidden[-1]  # (batch_size, hidden_dim)

        # Generate action probabilities
        action_logits = self.action_head(final_hidden)
        # Softmax
        action_probs = F.softmax(action_logits, dim=-1)
        # Check nan
        if torch.isnan(action_probs).any():
            print("NaN in softmax")
            print("action_logits:", action_logits)
            print("embedded_obs:", embedded_obs)
            print("obs_sequences:", obs_sequences)
            raise ValueError("NaN in action_probs")

        return action_probs, action_logits


class LazyObservableOperator:
    """
    Lazy computation with caching - only compute what you need, when you need it
    This is often the best approach for large state spaces
    """

    def __init__(self, env):
        self.env = env
        self.cache = {}
        self.state_to_idx = {state: i for i, state in enumerate(self.env.states)}

    def get_operator(self, obs_t, act_t):
        """Compute observable operator on-demand and cache results"""
        prod_pomdp_emiss = self.env.emiss

        key = (obs_t, act_t)
        if key not in self.cache:
            rows, cols, data = [], [], []

            # Only iterate over states that have transitions for this action
            for st_prime in self.env.states:
                if (st_prime not in self.env.transition or
                        act_t not in self.env.transition[st_prime]):
                    continue

                if (st_prime not in prod_pomdp_emiss or
                        act_t not in prod_pomdp_emiss[st_prime] or
                        obs_t not in prod_pomdp_emiss[st_prime][act_t]):
                    continue

                j = self.state_to_idx[st_prime]
                emiss_prob = prod_pomdp_emiss[st_prime][act_t][obs_t]

                if emiss_prob == 0:
                    continue

                # Only process reachable next states
                for st in self.env.transition[st_prime][act_t]:
                    trans_prob = self.env.transition[st_prime][act_t][st]
                    if trans_prob > 0:
                        i = self.state_to_idx[st]
                        value = trans_prob * emiss_prob
                        if value > 0:
                            rows.append(i)
                            cols.append(j)
                            data.append(value)

            # Create sparse matrix
            if data:
                oo = csr_matrix((data, (rows, cols)),
                                shape=(self.env.state_size, self.env.state_size))
            else:
                oo = csr_matrix((self.env.state_size, self.env.state_size))

            self.cache[key] = oo

        return self.cache[key]


class VariationalPolicyGradient:
    """
    Class for gradient calculation in multi-agent planning.
    Stores observable operator, sensor number, and fixed sensor list as attributes
    to avoid passing them to each function call.
    """

    def __init__(self, exp_name, env, experiment_num, tau, gamma, horizon, trajectories_num, iteration_num, step_size,
                 policy_network):
        """
        Initialize the gradient calculator with sensor configuration.
        """
        self.exp_name = exp_name
        self.env = env
        self.ex_num = experiment_num
        self.tau = tau  # temperature parameter
        self.gamma = gamma  # discounted factor
        self.T = horizon  # length of horizon
        self.M = trajectories_num
        self.iter_num = iteration_num
        self.eta = step_size
        # Initialize the observable operator using the ultra-fast method
        self.observable_operator = self.get_observable_operator_ultra_fast()
        # Initialize policy network as class attribute
        self.policy_net = policy_network
        # Define the optimizer for the policy network
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=self.eta)

    def obs_to_index(self, obs):
        """Convert observation to index for embedding"""
        # You may need to modify this based on your observation space
        try:
            return self.env.observations.index(obs)
        except:
            return 0  # Default to first observation if not found

    def pi_theta_network(self, policy_net, obs_sequence, a):
        """
        Get policy probability for action a given observation sequence
        Args:
            policy_net: Policy network
            obs_sequence: List of observations
            a: Action index
        Returns:
            Probability of taking action a
        """
        # Handle regular actions
        if a >= policy_net.action_size:
            return 0.0  # Invalid action index

        # Convert observations to indices
        if len(obs_sequence) == 0:
            obs_tensor = torch.tensor([[self.env.start_idx]], dtype=torch.long)
        else:
            obs_indices = [self.obs_to_index(obs) for obs in obs_sequence]
            obs_tensor = torch.tensor([obs_indices], dtype=torch.long)

        with torch.no_grad():
            action_probs, _ = policy_net(obs_tensor)
            return action_probs[0, a].item()

    def log_policy_gradient_network(self, policy_net, obs_sequence, a):
        """
        Compute log policy gradient for neural network
        Args:
            policy_net: Policy network
            obs_sequence: List of observations
            a: Action taken
        Returns:
            Gradient of log policy
        """
        # Convert observations to indices
        obs_indices = [self.obs_to_index(obs) for obs in obs_sequence]
        obs_tensor = torch.tensor([obs_indices], dtype=torch.long)

        # Forward pass
        action_probs, action_logits = policy_net(obs_tensor)

        # Compute log probability of taken action
        log_prob = torch.log(action_probs[0, a] + 1e-8)  # Add small epsilon for numerical stability

        # Compute gradients
        log_prob.backward()

        # Extract gradients (you might want to return them in a different format)
        gradients = []
        for param in policy_net.parameters():
            if param.grad is not None:
                gradients.append(param.grad.clone())

        return gradients

    def get_observable_operator_ultra_fast(self):
        """
        Ultra-fast implementation using vectorized operations and minimal memory
        """
        # Use lazy computation instead of precomputing everything
        print("Initializing lazy observable operator for efficient computation...")
        return LazyObservableOperator(self.env)

    def p_obs_g_actions(self, obs_list, act_list):
        """
        Modified to use class observable operator
        """
        mu_0 = self.env.initial_dist

        # Use the class observable operator
        oo = self.observable_operator.get_operator(obs_list[-1], act_list[-1])

        # Use sparse matrix operations
        one_vec = np.ones((1, self.env.state_size))
        probs = one_vec @ oo

        # Calculate the probability using sparse operations
        for t in reversed(range(len(obs_list) - 1)):
            oo = self.observable_operator.get_operator(obs_list[t], act_list[t])
            probs = probs @ oo

        # Final multiplication with initial distribution
        probs = probs @ mu_0

        # Handle both sparse and dense result formats
        if hasattr(probs, 'toarray'):
            return probs.toarray()[0, 0]
        elif hasattr(probs, 'shape') and len(probs.shape) > 0:
            return probs[0] if probs.shape == (1,) else probs[0, 0]
        else:
            return float(probs)

    def p_obs_g_actions_initial(self, obs_0, act_0):
        """
        Modified to use class observable operator
        """
        mu_0 = self.env.initial_dist

        # Use the class observable operator
        oo = self.observable_operator.get_operator(obs_0, act_0)

        # Use sparse matrix operations
        one_vec = np.ones((1, self.env.state_size))
        probs = one_vec @ oo
        probs = probs @ mu_0

        # Handle both sparse and dense result formats
        if hasattr(probs, 'toarray'):
            return probs.toarray()[0, 0]
        elif hasattr(probs, 'shape') and len(probs.shape) > 0:
            return probs[0] if probs.shape == (1,) else probs[0, 0]
        else:
            return float(probs)

    def p_theta_obs_g_context(self, obs_list, a_list):
        """
        Compute P(y, a_list; theta) for neural network policy
        Uses class observable operator and policy network

        a_list contains *action indices* (0,1,2,...)
        """
        # 1) Policy probability product (indices are correct here)
        policy_prod = self.pi_theta_network(self.policy_net, [], a_list[0])

        for i in range(len(obs_list) - 1):
            obs_history = obs_list[:i + 1]
            policy_prod *= self.pi_theta_network(
                self.policy_net,
                obs_history,
                a_list[i + 1],
            )

        # 2) Convert indices back to *symbols* for the POMDP part
        act_symbol_list = [self.env.actions[a_idx] for a_idx in a_list]
        act0_symbol = act_symbol_list[0]

        p_obs_g_acts_initial = self.p_obs_g_actions_initial(obs_list[0], act0_symbol)
        p_obs_g_acts = self.p_obs_g_actions(obs_list, act_symbol_list)
        # print("Initial", p_obs_g_acts_initial)
        # print("whole", p_obs_g_acts)

        if p_obs_g_acts_initial <= 0:
            return 0.0
        else:
            return (p_obs_g_acts / p_obs_g_acts_initial) * policy_prod

    def entopy_a_log_p_theta_obs(self, obs_list, a_list):
        # Make p_obs a tensor from the start
        p_obs = torch.zeros(1, dtype=torch.float32)
        temp_list = []
        for context in self.env.contexts:
            # update environment to the new context
            self.env.set_context(context)
            # ensure p_theta_obs_g_context returns a tensor
            p_context = self.p_theta_obs_g_context(obs_list, a_list)
            # context_distribution should be float or 0-D tensor
            weight = torch.tensor(
                self.env.context_distribution[context],
                dtype=torch.float32
            )
            temp_term = p_context * weight  # ~ p(obs, a, c)
            p_obs += temp_term  # sum_c p(obs, a, c)
            temp_list.append(temp_term)
        # Conditional entropy H(C | obs, a)
        # print(p_obs)
        entropy = torch.zeros(1, dtype=torch.float32)
        for i in range(len(self.env.contexts)):
            # p(c | obs, a) = p(obs, a, c) / p(obs, a)
            p_theta_context_g_obs = temp_list[i] / (p_obs + 1e-8)
            entropy += p_theta_context_g_obs * torch.log(p_theta_context_g_obs + 1e-8)
        entropy = -entropy  # H = -∑ p log p
        return entropy, torch.log(p_obs + 1e-8)

    def nabla_log_p_theta_obs_g_context(self, obs_list, a_list):
        # Clear previous grads
        self.policy_net.zero_grad()

        # Build computation graph for sum log π(a | history)
        total_log_prob = []

        # First action
        obs_tensor = torch.tensor([[self.env.start_idx]], dtype=torch.long)
        action_probs, _ = self.policy_net(obs_tensor)
        total_log_prob.append(torch.log(action_probs[0, a_list[0]] + 1e-8))

        # Subsequent actions
        for i in range(len(obs_list) - 1):
            obs_indices = [self.obs_to_index(obs) for obs in obs_list[:i + 1]]
            obs_tensor = torch.tensor([obs_indices], dtype=torch.long)
            action_probs, _ = self.policy_net(obs_tensor)
            total_log_prob.append(torch.log(action_probs[0, a_list[i + 1]] + 1e-8))

        # Sum log-probabilities
        total_log_prob = torch.stack(total_log_prob).sum()

        # BACKPROP HERE
        total_log_prob.backward()

        # Extract gradients
        gradients = []
        for param in self.policy_net.parameters():
            if param.grad is None:
                gradients.append(torch.zeros_like(param))
            else:
                gradients.append(param.grad.clone())

        return gradients

    def entropy_a_grad_network(self, context_data, reward_data, obs_data, act_data):
        """
        Compute entropy and gradients for neural network policy
        Uses class observable operator
        """
        M = len(obs_data)
        avg_entropy = torch.zeros(1, dtype=torch.float32)
        avg_reward = torch.zeros(1, dtype=torch.float32)
        avg_value = torch.zeros(1, dtype=torch.float32)
        # Accumulate gradients
        total_grad_H = None

        for k in range(M):
            obs_list_k = obs_data[k]
            act_list_k = act_data[k]
            a_list_k = [self.env.actions.index(act) for act in act_list_k]
            context = context_data[k]
            traj_reward = reward_data[k]

            entropy, log_P_theta_yk = self.entopy_a_log_p_theta_obs(obs_list_k, a_list_k)  # independent from context
            print("entropy", entropy)
            print("traj_reward", traj_reward)
            avg_entropy += entropy
            avg_reward += traj_reward

            # reset the context
            self.env.set_context(context)
            reward_term = torch.tensor(traj_reward / self.tau, dtype=torch.float32)
            # Get gradients for this trajectory
            grad_log_P_theta_yk_g_context = self.nabla_log_p_theta_obs_g_context(obs_list_k, a_list_k)
            constant = log_P_theta_yk - reward_term
            avg_value += traj_reward - self.tau * entropy

            # Accumulate gradients
            if total_grad_H is None:
                total_grad_H = [constant * grad.clone() for grad in grad_log_P_theta_yk_g_context]
            else:
                for i, grad in enumerate(grad_log_P_theta_yk_g_context):
                    total_grad_H[i] += constant * grad

        # Average value
        avg_value = avg_value / M  # This is not trajectory estimate just average.
        avg_entropy = avg_entropy / M
        avg_reward = avg_reward / M
        # Average gradients
        nabla = [grad / M for grad in total_grad_H]

        return avg_entropy.item(), avg_reward.item(), avg_value.item(), nabla

    def action_sampler_network(self, policy_net, obs_sequence):
        """
        Sample action from neural network policy
        Args:
            obs_sequence: List of observations
            is_last_step: Whether this is the last step (when 'end' action should be chosen)
        Returns:
            Sampled action
            :param obs_sequence:
            :param policy_net:
        """
        # For non-last steps, sample from the policy network (excluding end action)
        if len(obs_sequence) == 0:
            obs_tensor = torch.tensor([[self.env.start_idx]], dtype=torch.long)  # Dummy observation for empty sequence
        else:
            obs_indices = [self.obs_to_index(obs) for obs in obs_sequence]
            obs_tensor = torch.tensor([obs_indices], dtype=torch.long)

        with torch.no_grad():
            action_probs, _ = policy_net(obs_tensor)
            probs = action_probs[0].cpu().numpy()

        action_idx = np.random.choice(len(probs), p=probs)
        return self.env.actions[action_idx]

    def sample_data(self, M, T):
        context_data = []
        st_data = []
        act_data = []
        obs_data = []
        reward_data = []
        for m in range(M):
            st_list = []
            act_list = []
            obs_list = []
            traj_reward = 0
            context = choice(self.env.contexts)
            self.env.set_context(context)
            st = self.env.initial_states[0]
            st_list.append(st)
            reward = self.env.reward_sampler(st)
            # print(reward)
            traj_reward += reward
            # Sample sensing action
            act = self.action_sampler_network(self.policy_net, obs_list)
            act_list.append(act)
            # Get the observation of initial state
            obs = self.env.observation_function_sampler(st, act)
            obs_list.append(obs)

            for t in range(T - 1):
                # sample the next state
                st = self.env.next_state_sampler(st, act)
                st_list.append(st)
                reward = self.env.reward_sampler(st)
                # print(reward)
                traj_reward += reward
                # Sample sensing action
                act = self.action_sampler_network(self.policy_net, obs_list)
                act_list.append(act)
                # Add the observation
                obs = self.env.observation_function_sampler(st, act)
                obs_list.append(obs)

            context_data.append(context)
            st_data.append(st_list)
            obs_data.append(obs_list)
            act_data.append(act_list)
            reward_data.append(traj_reward)
        return context_data, st_data, obs_data, act_data, reward_data

    def train(self):
        # Create directory, skip if exists
        os.makedirs(f"data_{self.exp_name}", exist_ok=True)
        os.makedirs(f"data_{self.exp_name}/Values", exist_ok=True)
        os.makedirs(f"data_{self.exp_name}/Graphs", exist_ok=True)

        # Storage lists
        value_list = []
        entropy_list = []
        reward_list = []

        # Training loop
        for i in range(self.iter_num):
            start = time.time()

            # Sample trajectories
            context_data, st_data, obs_data, act_data, reward_data = \
                self.sample_data(self.M, self.T)

            # Compute gradients
            entropy, traj_reward, avg_value, grad = \
                self.entropy_a_grad_network(context_data, reward_data, obs_data, act_data)

            # --------------------
            # Logging
            # --------------------
            print(f"Iteration {i + 1}/{self.iter_num}")
            print("  Average value:   ", avg_value)
            print("  Average entropy: ", entropy)
            print("  Avg reward/traj: ", traj_reward)
            # print(grad)
            # print(obs_data)

            value_list.append(avg_value)
            entropy_list.append(entropy)
            reward_list.append(traj_reward)

            # --------------------
            # Gradient update
            # --------------------
            self.optimizer.zero_grad()
            self.apply_gradients(grad)
            self.optimizer.step()

            end = time.time()
            print(f"Iteration {i + 1} done in {end - start:.2f} s")
            print("#" * 100)

        # -------------------------------------------------------
        # Save policy and logged values
        # -------------------------------------------------------
        with open(f'./data_{self.exp_name}/Values/policy_net_{self.ex_num}.pkl', "wb") as f:
            torch.save(self.policy_net.state_dict(), f)

        with open(f'./data_{self.exp_name}/Values/value_{self.ex_num}.pkl', "wb") as f:
            pickle.dump(value_list, f)

        with open(f'./data_{self.exp_name}/Values/entropy_{self.ex_num}.pkl', "wb") as f:
            pickle.dump(entropy_list, f)

        with open(f'./data_{self.exp_name}/Values/reward_{self.ex_num}.pkl', "wb") as f:
            pickle.dump(reward_list, f)

        # -------------------------------------------------------
        # Plot: Value curve
        # -------------------------------------------------------
        iters = range(self.iter_num)

        plt.figure()
        plt.plot(iters, value_list, label='Average Value')
        plt.xlabel("Iteration")
        plt.ylabel("Value")
        plt.legend()
        plt.title("Average Value per Iteration")
        plt.savefig(f'./data_{self.exp_name}/Graphs/Ex_{self.ex_num}_value.png')
        plt.show()

        # -------------------------------------------------------
        # Plot: Entropy curve
        # -------------------------------------------------------
        plt.figure()
        plt.plot(iters, entropy_list, label='Entropy', color='orange')
        plt.xlabel("Iteration")
        plt.ylabel("Entropy")
        plt.title("Trajectory Entropy per Iteration")
        plt.legend()
        plt.savefig(f'./data_{self.exp_name}/Graphs/Ex_{self.ex_num}_entropy.png')
        plt.show()

        # -------------------------------------------------------
        # Plot: Reward curve
        # -------------------------------------------------------
        plt.figure()
        plt.plot(iters, reward_list, label='Average Reward', color='green')
        plt.xlabel("Iteration")
        plt.ylabel("Reward")
        plt.title("Average Reward per Iteration")
        plt.legend()
        plt.savefig(f'./data_{self.exp_name}/Graphs/Ex_{self.ex_num}_reward.png')
        plt.show()

        print("Training finished. Plots saved.")

    def apply_gradients(self, gradients):
        """
        Apply computed gradients to the policy network
        """
        for param, grad in zip(self.optimizer.param_groups[0]['params'], gradients):
            if param.grad is None:
                param.grad = grad
            else:
                param.grad += grad


def main():
    # Set random seed for reproducibility
    sd = 1337
    horizon = 10
    set_random_seeds(sd)

    # Name the environment
    exp_name = 'line_grid_1'

    # Initialize the training process
    env = Environment(choice([0, 1]), 0.1, 0.1)
    env.seed(sd)
    # Initialize policy network
    policy_net = PolicyNetwork(obs_vocab_size=len(env.observations), action_size=env.action_size,
                               hidden_dim=64, max_seq_len=horizon)

    # Get the trainer
    trainer = VariationalPolicyGradient(exp_name=exp_name, env=env, experiment_num=3, tau=0.1, gamma=1, horizon=horizon,
                                        trajectories_num=20, iteration_num=1000, step_size=0.001,
                                        policy_network=policy_net)

    # Train the selected perception agent
    trainer.train()


if __name__ == "__main__":
    main()
