# Report

## Learning Algorithm

The code uses Deep Q-Learning Algorithm for training the model. The implemenation is wrapped within the Agent class (i.e., `agent.py`)


### Model architecture

Model architecture has been chosen based on the mean per 100 sequential episode scores.

It evolved from sequentially connected 3 dense/linear layers.

- L1: <state_size x 64>
- L2: <64 x 64>
- L3 :<64 x action_size>

The score seemed to hit a ceiling around 13-14 range after couple of hundred episodes. I suspected it was underfitting for the current problem at hand. So, I expanded model learnable parameters.

First try was expanding it to the 5 layer model with ...

- L1: <state_size x 256>
- L2: <256 x 256>
- L3: <256 x 128>
- L4: <128 x 64>
- L5 :<64 x action_size>

The score managed to achieve 14-16 band. Yet, it the score started to decrement to the 13 after couple of hundred episodes. I suspected it was overfitting for the current problem at hand. So, I tried something in between.

Saved model consist of sequentially connected 4 dense/linear layers.

- L1: <state_size x 128>
- L2: <128 x 128>
- L3: <128 x 64>
- L4 :<64 x action_size>

### Agent

Agent utilizes 2 models for the DQN network. i.e., local (w), target (w-). Project uses the following hyperparameters during training

- LEARNING_RATE = 5e-4
- GAMMA = 0.99
- TAU = 1e-3

- BUFFER_SIZE = int(1e5)
- BATCH_SIZE = 64
- UPDATE_EVERY = 4


#### Act

Agent acts with an e-greedy algorithm. It either acts randomly or using the local model with frozen weights (no grad.).

The agent records each experience (i.e., `<S, A, R, S'>`) to the replay buffer with each step. With every `UPDATE_EVERY` steps, the agent tries to learn from experiences stored in the replay buffer.


#### Learn

The replay buffer randomly samples a given `BATCH_SIZE` experiences (i.e., `<S, A, R, S'>`) for learning after each `UPDATE_EVERY` itearation.

The agent makes prediction for each given experience with both local and target models. Then, the gradients are applied to the local model with MSE loss between both predictions.

Finally, the target model parameters are updates based on the `TAU` variable.


## Plot of Rewards
![alt text][logo]

[logo]: ./scores.png "Plot of Rewards"

## Ideas for Future Work

It is possible to improve the model by using dropout layers to prevent overfitting at latter stages.

There are also various methods to improve Q-learning, where we use ...

- Reward clipping
- Error clipping
- etc...


