from src.memory import Memory
import random
import torch


def create_transitions(state=None,
                       action=None,
                       reward=None,
                       next_state=None,
                       finish=None,
                       n_state=None):
    n_state = 4 if n_state is None else n_state
    state = [random.random() for x in range(n_state)] if state is None else state
    action = random.sample([0, 1], 1)[0] if action is None else action
    reward = random.random() if reward is None else reward
    next_state = [random.random() for x in range(n_state)] if next_state is None else next_state
    finish = random.sample([False, True], 1)[0] if finish is None else finish
    return state, action, reward, next_state, finish


def test_init():
    memory = Memory(10, 4)
    assert len(memory) == 0
    assert not memory.is_memory_full()
    assert memory.position == 0
    assert memory.n_state == 4
    assert memory.capacity == 10


def test_len():
    memory = Memory(10,4)
    for idx in range(5):
        state, action, reward, next_state, finish = create_transitions()
        memory.save(state, action, reward, next_state, finish)
    assert len(memory) == 5


def test_is_memory_full():
    memory = Memory(10, 4)
    assert not memory.is_memory_full()
    for idx in range(5):
        state, action, reward, next_state, finish = create_transitions()
        memory.save(state, action, reward, next_state, finish)
    assert not memory.is_memory_full()
    for idx in range(5):
        state, action, reward, next_state, finish = create_transitions()
        memory.save(state, action, reward, next_state, finish)
    assert memory.is_memory_full()
    for idx in range(5):
        state, action, reward, next_state, finish = create_transitions()
        memory.save(state, action, reward, next_state, finish)
    assert memory.is_memory_full()


def test_update_postiion():
    memory = Memory(10, 4)
    assert memory.position == 0
    for idx in range(5):
        state, action, reward, next_state, finish = create_transitions()
        memory.save(state, action, reward, next_state, finish)
    assert memory.position == 5
    for idx in range(5):
        state, action, reward, next_state, finish = create_transitions()
        memory.save(state, action, reward, next_state, finish)
    assert memory.position == 0
    for idx in range(5):
        state, action, reward, next_state, finish = create_transitions()
        memory.save(state, action, reward, next_state, finish)
    assert memory.position == 5


def test_sample():
    memory = Memory(10, 4)
    for idx in range(5):
        state, action, reward, next_state, finish = create_transitions(
            state = [idx, idx, idx, idx],
            reward = idx,
            next_state = [idx, idx, idx, idx]
        )
        memory.save(state, action, reward, next_state, finish)
    sampled_state, sampled_action, sampled_reward, sampled_next_state, sampled_finish = memory.sample(3)
    assert sampled_state.shape == (3, 4)
    assert sampled_next_state.shape == (3, 4)
    assert sampled_state[:, 0] == sampled_reward

    tensor_state, tensor_action, tensor_reward, tensor_next_state, tensor_finish = memory.sample(3, return_tensor=True)
    assert tensor_state.shape == torch.Size([3, 4])
    assert tensor_next_state.shape == torch.Size([3, 4])

    assert tensor_state.type() == 'torch.cuda.FloatTensor'
    assert tensor_action.type() == 'torch.cuda.IntTensor'
    assert tensor_reward.type() == 'torch.cuda.FloatTensor'
    assert tensor_next_state.type() == 'torch.cuda.FloatTensor'
    assert tensor_finish.type() == 'torch.cuda.IntTensor'

