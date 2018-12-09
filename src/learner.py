import torch


class Optimize_Molel(object):
    def __init__(self, old_qnet, new_qnet, optimizer):
        self.new_qnet = new_qnet
        self.old_qnet = old_qnet
        self.old_qnet.load_state_dict(self.old_qnet.state_dict())
        self.old_qnet = self.old_qnet.eval()
        self.optimizer = optimizer
        self.torch_device = self.config['DEVICE']
        self.gamma =self.config['GAMMA']
        self.batch_size = self.config['BATCH_SIZE']

    def calc_q(self, state_tensor):
        with torch.no_grad():
            return self.old_qnet(state_tensor).cpu().numpy()

    def optimize_new_qnet(self, memory, config):
        torch_device = config.device
        gamma = config.gamma
        batch_size = config.batch_size


        if len(memory) < batch_size:
            pass
        else:
            state_tensor, action_tensor, reward_tensor, next_state_tensor, finish_tensor = memory.sample(
                batch_size, return_tensor=True, torch_device=torch_device
            )

            finish_index = torch.nonzero(finish_tensor.view(-1)).view(-1)
            cur_q = self.new_qnet(state_tensor)
            cur_qa = cur_q.gather(1, action_tensor)

            with torch.no_grad():
                unfinished_next_state_tensor = next_state_tensor[finish_index, :]
                next_q = self.old_net(unfinished_next_state_tensor)
                next_qa = next_q.max(1)[0]
                exp_qa = reward_tensor + (gamma * next_qa)

            # LOSS FUNCTION

        raise NotImplementedError()
