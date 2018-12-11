# Pytorch DQN

An attempt to create a modularized, Deep Q Network which uses Q-learning method to perform reinforcement learning on a given environment

## Component
#### Arch
Pytorch nn.Module which is responsible in approximating value function given the environment state
#### Cacher
Debugging purpose - caching running metrics (number of episode, loss value) for us to check the progress of the reinforcement learning
#### Config
All config values goes to here
#### Envs
Environment which agent is supposed to learn. Extend openai.gym objects
#### Learner
Performs the Q-learning update, given the saved transitions within the memory
#### Memory
Memorize the past transitions to be used by `Learner` to perform the Q-Learning update
#### Policy
Performs the Policy of choosing actions, based on a given value function
#### Agent
Wrapper functions for all component parts, perform learning, and playing within the environment