import matplotlib.pyplot as plt
import seaborn as sns


class CacherObject:
    def __init__(self, name, n):
        self.name = name
        self.n = n
        self.memory = []

    def reset_memory(self):
        self.memory = []

    def save(self, input_value):
        if self.n == 1:
            item = [input_value]
        else:
            assert self.n == len(input_value)
            item = [x for x in input_value]
        self.memory.append(item)


class Cacher:
    def __init__(self):
        self.cacher_dict = dict()

    def new_cacher(self, name, n):
        if name in self.cacher_dict.keys():
            raise ValueError('CacherObject already exist')
        cacherobject = CacherObject(name, n)
        self.cacher_dict[name] = cacherobject

    def save_cacher(self, name, input_value):
        self.cacher_dict[name].save(input_value)

    def reset_cacher(self, name):
        self.cacher_dict[name].reset_memory()

    def plot_cacher(self, name):
        assert self.cacher_dict[name].n == 1
        value = self.cacher_dict[name].memory
        flat_value = [item for sublist in value for item in sublist]
        sns.lineplot(range(len(flat_value)), flat_value).set_title(name)
        plt.show()
