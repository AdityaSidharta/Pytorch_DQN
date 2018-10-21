import torch


# TODO finish Q-Function
class Q_Function(object):
    def __init__(self, old_qnet, new_qnet, optimizer):
        self.new_qnet = new_qnet
        self.old_qnet = old_qnet
        self.old_qnet.load_state_dict(self.old_qnet.state_dict())
        self.old_qnet = self.old_qnet.eval()
        self.optimizer = optimizer

    def calc_q(self, state_tensor):
        with torch.no_grad():
            return self.old_qnet(state_tensor).cpu().numpy()

    def optimize_new_qnet(self, batch_size, memory, config):
        if len(memory) < batch_size:
            pass
        else:
            state_tensor, action_tensor, reward_tensor, next_state_tensor, finish_tensor = memory.sample(
                batch_size, torch=True, device=config["DEVICE"]
            )

            cur_q = self.new_qnet(state_tensor)
            cur_qa = cur_q.gather(action_tensor)

            n_actions = cur_q.shape[1]

            with torch.no_grad():
                unfinished_next_state_tensor = next_state_tensor[~finish_tensor]
                next_q = torch.zeros_like(cur_q)
                next_q[~finish_tensor] = old_net(unfinished_next_state_tensor)
                next_qa = next_q.max(1)[0]

                exp_qa = reward_tensor + (config["GAMMA"] * next_qa)
                # LOSS FUNCTION

        raise NotImplementedError()
