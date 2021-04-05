from unityagents import UnityEnvironment
import numpy as np
import torch
from collections import deque

from agent import Agent

env = UnityEnvironment(file_name="/Applications/Banana.app", no_graphics=True)

# get the default brain
brain_name = env.brain_names[0]
brain = env.brains[brain_name]

# reset the environment
env_info = env.reset(train_mode=True)[brain_name]

# number of agents in the environment
print("Number of agents:", len(env_info.agents))

# number of actions
action_size = brain.vector_action_space_size
print("Number of actions:", action_size)

# examine the state space
state = env_info.vector_observations[0]
print("States look like:", state)
state_size = len(state)
print("States have length:", state_size)

print(env_info.vector_observations)


def train(
    env,
    agent,
    brain_name,
    n_episodes=2000,
    max_t=1000,
    eps_start=1.0,
    eps_end=0.01,
    eps_decay=0.995,
):

    scores = []
    scores_window = deque(maxlen=100)
    eps = eps_start

    model_score = 13.0

    for i_episode in range(n_episodes):
        env_info = env.reset(train_mode=True)[brain_name]  # reset the environment
        state = env_info.vector_observations[0]  # get the current state
        score = 0  # initialize the score

        for t in range(max_t):
            action = agent.act(state, eps)

            env_info = env.step(action)[
                brain_name
            ]  # send the action to the environment
            next_state = env_info.vector_observations[0]  # get the next state
            reward = env_info.rewards[0]  # get the reward
            done = env_info.local_done[0]  # see if episode has finished

            agent.step(state, action, reward, next_state, done)
            state = next_state
            score += reward

            if done:
                break

        scores.append(score)
        scores_window.append(score)
        eps = max(eps_end, eps * eps_decay)
        mean_score = np.mean(scores_window)

        if i_episode % 100 == 0:
            print("\rEpisode {}\tAverage Score: {:.2f}".format(i_episode, mean_score))
            if mean_score >= model_score:
                model_score = mean_score
                print(
                    "\nEnvironment solved better in episode {:d}!\tAverage Score: {:.2f}".format(
                        i_episode - 100, np.mean(scores_window)
                    )
                )
                torch.save(agent.q_local.state_dict(), "checkpoint.pth")
                print("Saved model with mean score: {}".format(mean_score))

    print("Completed training")
    return scores


device = "cuda" if torch.cuda.is_available() else "cpu"
agent = Agent(state_size, action_size, device)

scores = train(env, agent, brain_name)
print("Scores: \n{}".format(scores))

env.close()
