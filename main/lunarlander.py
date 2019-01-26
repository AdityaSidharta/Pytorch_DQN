import argparse
import os

import torch.optim as optim

from src.agent import *
from src.arch import *
from src.cacher import *
from src.config import *
from src.envs import *
from src.learner import *
from src.memory import *
from src.policy import *
from utils.envs import output_path


def run_cartpole(
    batch_size,
    gamma,
    eps_start,
    eps_end,
    eps_decay,
    n_target_update,
    n_episode,
    device,
    memory_space,
    observation_space,
    action_space,
):
    arch = LunarNet(1000).to(DEVICE)
    config = Config(
        batch_size,
        gamma,
        eps_start,
        eps_end,
        eps_decay,
        n_target_update,
        n_episode,
        device,
        memory_space,
        observation_space,
        action_space,
    )
    envs = gym.make("LunarLander-v2").unwrapped
    learner = Learner(arch, optim.RMSprop)
    memory = Memory(memory_space, observation_space)
    policy = EGreedy()
    cacher = Cacher()
    agent = Agent(learner, memory, policy, envs, config, cacher)

    agent.train_agent(n_episode)
    agent.play_agent(10)

    agent.cacher.plot_cacher(
        "train_reward",
        filename=os.path.join(output_path, "lunarlander_train_reward.png"),
    )
    agent.cacher.plot_cacher(
        "play_reward", filename=os.path.join(output_path, "lunarlander_play_reward.png")
    )


if __name__ == "__main__":
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    OBSERVATION_SPACE = (8,)
    ACTION_SPACE = 4

    parser = argparse.ArgumentParser(description="Config file for Cartpole")
    parser.add_argument(
        "--n_episode", type=int, default=50000, help="Number of Training Episodes"
    )
    parser.add_argument(
        "--batch_size", type=int, default=256, help="Number of Batch Size in Training"
    )
    parser.add_argument(
        "--gamma",
        type=float,
        default=0.999,
        help="Discounting Factor for Reward function",
    )
    parser.add_argument(
        "--eps_start", type=float, default=0.95, help="Start value for Epsilon Greedy"
    )
    parser.add_argument(
        "--eps_end", type=float, default=0.05, help="End value for Epsilon Greedy"
    )
    parser.add_argument(
        "--eps_decay",
        type=int,
        default=45000,
        help="Number of Episodes Epsilon Decay is performed",
    )
    parser.add_argument(
        "--n_target_update",
        type=int,
        default=10,
        help="Interval where Target Net is updated",
    )
    parser.add_argument(
        "--memory_space", type=int, default=1000000, help="Size of Memory Space"
    )
    args = parser.parse_args()

    run_cartpole(
        args.batch_size,
        args.gamma,
        args.eps_start,
        args.eps_end,
        args.eps_decay,
        args.n_target_update,
        args.n_episode,
        DEVICE,
        args.memory_space,
        OBSERVATION_SPACE,
        ACTION_SPACE,
    )
