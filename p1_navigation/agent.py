import numpy as np
import random
from collections import namedtuple, deque
from model import QNetwork
import torch
import torch.nn.functional as F

LEARNING_RATE = 5e-4
GAMMA = 0.99
TAU = 1e-3

BUFFER_SIZE = int(1e5)
BATCH_SIZE = 64
UPDATE_EVERY = 4


class Agent:
    def __init__(self, state_size, action_size, device):
        self.device = device
        self.q_local = QNetwork(state_size, action_size)
        self.q_target = QNetwork(state_size, action_size)
        self.optimizer = torch.optim.Adam(self.q_local.parameters(), lr=LEARNING_RATE)

        self.replay_buffer = ReplayBuffer(BUFFER_SIZE, BATCH_SIZE)

        self.action_size = action_size
        self.step_count = 0

    def step(self, state, action, reward, next_state, done):
        self.replay_buffer.add(state, action, reward, next_state, done)

        self.step_count = (self.step_count + 1) % UPDATE_EVERY
        if self.step_count == 0:
            # If enough samples are available in memory, get random subset and learn
            if len(self.replay_buffer) > BATCH_SIZE:
                experiences = self.replay_buffer.sample(self.device)
                self.learn(experiences, GAMMA)

    def act(self, state, eps=0.0):
        """ Given the state, select an action.
        
        Params
        ======
        - state: the current state of the environment
        
        Returns
        ======
        - action: an integer, compatible with the task's action space
        """
        # TODO: Breakdown the following
        state = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
        self.q_local.eval()  # Mark layers as trainable=false. e.g., no dropout

        with torch.no_grad():
            action_values = self.q_local(state)
        self.q_local.train()

        if random.random() > eps:
            return np.argmax(action_values.cpu().data.numpy())
        else:
            return random.choice(np.arange(self.action_size))

    def learn(self, experiences, gamma):
        """ Update the agent's knowledge, using the most recently sampled tuple.
        
        Params
        ======
        - state: the previous state of the environment
        - action: the agent's previous choice of action
        - reward: last reward received
        - next_state: the current state of the environment
        - done: whether the episode is complete (True or False)
        """

        states, actions, rewards, next_states, dones = experiences

        # TODO: Unpack
        # Get max predicted Q values (for next states) from target model
        Q_targets_next = self.q_target(next_states).detach().max(1)[0].unsqueeze(1)
        # Compute Q targets for current states
        Q_targets = rewards + (gamma * Q_targets_next * (1 - dones))

        # Get expected Q values from local model
        Q_expected = self.q_local(states).gather(1, actions)

        loss = F.mse_loss(Q_expected, Q_targets)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # TODO: ???
        TAU = 1e-3  # for soft update of target parameters
        self.soft_update(self.q_local, self.q_target, TAU)

    def soft_update(self, local_model, target_model, tau):
        """ Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target

        Params
        ======
            local_model (PyTorch model): weights will be copied from
            target_model (PyTorch model): weights will be copied to
            tau (float): interpolation parameter 
        """
        for target_param, local_param in zip(
            target_model.parameters(), local_model.parameters()
        ):
            target_param.data.copy_(
                tau * local_param.data + (1.0 - tau) * target_param.data
            )


class ReplayBuffer:
    def __init__(self, buffer_size, batch_size):
        self.memory = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        self.experience = namedtuple(
            "Experience",
            field_names=["state", "action", "reward", "next_state", "done"],
        )

    def add(self, state, action, reward, next_state, done):
        """ Add experience to memory """
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)

    def sample(self, device):
        """ Randomly sample a batch from memory """
        experiences = random.sample(self.memory, k=self.batch_size)

        states = (
            torch.from_numpy(np.vstack([e.state for e in experiences if e is not None]))
            .float()
            .to(device)
        )
        actions = (
            torch.from_numpy(
                np.vstack([e.action for e in experiences if e is not None])
            )
            .long()
            .to(device)
        )
        rewards = (
            torch.from_numpy(
                np.vstack([e.reward for e in experiences if e is not None])
            )
            .float()
            .to(device)
        )
        next_states = (
            torch.from_numpy(
                np.vstack([e.next_state for e in experiences if e is not None])
            )
            .float()
            .to(device)
        )
        dones = (
            torch.from_numpy(
                np.vstack([e.done for e in experiences if e is not None]).astype(
                    np.uint8
                )
            )
            .float()
            .to(device)
        )

        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        """ Return the current size of the internal memory """
        return len(self.memory)
