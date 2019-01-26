# Pytorch DQN

Yet another attempt to create a modularized, DQN Reinforcment 
learning framework which is written in Pytorch.

## Instalation
```
make setup
source config.sh
```

## Usage
To use this project, you should initialize the following object
- Arch : Pytorch `nn.module` which takes in the current state as an input, and provides value function for every action 
- Config : Class containing all of the config variables. See `main` for example
- Envs : `gym.envs` Object
- Learner: Learner object, containing `arch` and `torch.optim`
- Memory : Memory object, giving the memory space and observation space as the input
- policy : Type of Policy class used, e.g `EGreedy` 
- cacher : Cacher object
- Agent : Agent Object that takes in `learner`, `memory`, `policy`, `envs`, `config`, `cacher`

Then, we can use `agent.train_agent()` to perform the DQN training. After training, `agent.play_agent(n_plays)`
can be used to check the performance of the model

Two examples is provided in the repo, the DQN implementation for `CartPole` problem and `LunarLander` problem. Both of this implementation is located at the `main` folder

### CartPole
```
python main/cartpole.py (--batchsize --n_episode --gamma --eps_start --eps_end --eps_decay --n_target_update --memory_space)
```
All of the argument is optional.

`cartpole_play_reward.png` and `cartpole_train_reward.png` will be created under output folder to show the training and playing result
### LunarLander
```
python main/lunarlander.py (--batchsize --n_episode --gamma --eps_start --eps_end --eps_decay --n_target_update --memory_space)
```
All of the argument is optional.

`lunarlander_play_reward.png` and `lunarlander_train_reward.png` will be created under output folder to show the training and playing result

## Component
### Arch
Pytorch nn.Module which is responsible in approximating value function given the environment state
### Cacher
Debugging purpose - caching running metrics (number of episode, loss value) for us to check the progress of the reinforcement learning
### Config
All config values goes to here
### Envs
Environment which agent is supposed to learn. Extend openai.gym objects
### Learner
Performs the Q-learning update, given the saved transitions within the memory
### Memory
Memorize the past transitions to be used by `Learner` to perform the Q-Learning update
### Policy
Performs the Policy of choosing actions, based on a given value function
### Agent
Wrapper functions for all component parts, perform learning, and playing within the environment

## License
See the [LICENSE.md](LICENSE.md) file for details