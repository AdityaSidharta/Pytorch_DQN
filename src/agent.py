from tqdm import tqdm
from utils.logger import log


class Agent(object):
    def __init__(self, learner, memory, policy, envs, config, cacher):
        self.learner = learner
        self.memory = memory
        self.policy = policy
        self.envs = envs
        self.config = config
        self.cacher = cacher

        self.cacher.new_cacher("train_reward", 1)
        self.cacher.new_cacher("play_reward", 1)
        self.cacher.new_cacher("loss", 1)

    def train_episode(self):
        cur_state = self.envs.reset()
        done = False
        total_reward = 0
        while not done:
            value_function = self.learner.predict(cur_state, self.config)
            chosen_action = self.policy.select_action(value_function, self.config)
            nxt_state, reward, done, _ = self.envs.step(chosen_action)
            self.memory.save(cur_state, chosen_action, reward, nxt_state, done)
            cur_state = nxt_state
            self.learner.learn(self.memory, self.config, self.cacher)
            total_reward = total_reward + reward
        return total_reward

    def train_agent(self, n_episode=None):
        n_target_update = self.config.n_target_update
        n_episode = self.config.n_episode if not n_episode else n_episode
        for idx in tqdm(range(n_episode)):
            total_reward = self.train_episode()
            log.debug("Number of episode reward: {}".format(total_reward))
            self.cacher.save_cacher("train_reward", total_reward)
            if idx % n_target_update == 0:
                self.learner.update()
        self.learner.update()

    def play_episode(self):
        cur_state = self.envs.reset()
        done = False
        total_reward = 0
        while not done:
            value_function = self.learner.predict(cur_state, self.config)
            chosen_action = self.policy.best_action(value_function, self.config)
            nxt_state, reward, done, _ = self.envs.step(chosen_action)
            self.memory.save(cur_state, chosen_action, reward, nxt_state, done)
            cur_state = nxt_state
            self.envs.render()
            total_reward = total_reward + reward
        self.envs.close()
        return total_reward

    def play_agent(self, n_play):
        for _ in tqdm(range(n_play)):
            total_reward = self.play_episode()
            self.cacher.save_cacher("play_reward", total_reward)
            log.debug("Number of episode reward: {}".format(total_reward))
