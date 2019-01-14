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

        self.cacher.new_cacher('train_eps', 1)
        self.cacher.new_cacher('loss', 1)

    def train_episode(self):
        cur_state = self.envs.reset()
        done = False
        count = 0
        while not done:
            count = count + 1
            max_action = self.learner.predict(cur_state, self.config)
            chosen_action = self.policy.select_action(max_action, self.config)
            nxt_state, reward, done, _ = self.envs.step(chosen_action)
            self.memory.save(cur_state, chosen_action, reward, nxt_state, done)
            cur_state = nxt_state
            self.learner.learn(self.memory, self.config, self.cacher)
        log.debug('Number of train episode : {}'.format(count))
        self.cacher.save_cacher('train_eps', count)

    def train_agent(self, n_episode):
        n_target_update = self.config.n_target_update
        for idx in tqdm(range(n_episode)):
            self.train_episode()
            if idx % n_target_update == 0:
                self.learner.update()

    def play_episode(self):
        cur_state = self.envs.reset()
        done = False
        count = 0
        while not done:
            count = count + 1
            value_function = self.learner.predict(cur_state, self.config)
            action = self.policy.best_action(value_function, self.config)
            _, _, done, _ = self.envs.step(action)
            self.envs.render()
        log.debug('Number of play episode : {}'.format(count))
