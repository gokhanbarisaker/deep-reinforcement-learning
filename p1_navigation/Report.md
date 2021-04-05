# Report

## Learning Algorithm

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

Project uses the following hyperparameters during training

- LEARNING_RATE = 5e-4
- GAMMA = 0.99
- TAU = 1e-3

- BUFFER_SIZE = int(1e5)
- BATCH_SIZE = 64
- UPDATE_EVERY = 4


## Plot of Rewards
![alt text][logo]

[logo]: ./scores.png "Plot of Rewards"

## Ideas for Future Work

It is possible to improve the model by using dropout layers to prevent overfitting at latter stages.

There are also various methods to improve Q-learning, where we use ...

- Reward clipping
- Error clipping
- etc...


