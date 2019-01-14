import numpy as np
import torch
import torch.nn.functional as F
from utils.logger import log


class Learner:
    def __init__(self, arch, optimizer):
        self.new_qnet = arch
        self.old_qnet = arch
        self.old_qnet.load_state_dict(self.new_qnet.state_dict())
        self.old_qnet = self.old_qnet.eval()
        self.optimizer = optimizer(self.new_qnet.parameters())

    def predict(self, state, config):
        with torch.no_grad():
            state_tensor = torch.from_numpy(state).to(config.device, dtype=torch.float)
            state_tensor = state_tensor.view(1, -1)
            value = self.new_qnet(state_tensor).cpu().numpy().squeeze()
            if config.action_type == 'DISCRETE':
                action = np.argmax(value)
            else:
                raise NotImplementedError()
            return action

    def learn(self, memory, config, cacher):
        torch_device = config.device
        gamma = config.gamma
        batch_size = config.batch_size

        if len(memory) < batch_size:
            pass
        else:
            state_tensor, action_tensor, reward_tensor, next_state_tensor, finish_tensor = memory.sample(
                batch_size, return_tensor=True, torch_device=torch_device
            )

            unfinish_idx = (finish_tensor == 0).nonzero().view(-1).view(-1)
            cur_q = self.new_qnet(state_tensor)
            cur_qa = cur_q.gather(1, action_tensor.view(-1, 1)).view(-1)

            with torch.no_grad():
                unfinish_next_state_tensor = next_state_tensor[unfinish_idx, :]
                next_q = self.old_qnet(unfinish_next_state_tensor)
                next_qa = torch.zeros_like(reward_tensor)
                next_qa[unfinish_idx] = next_q.max(1)[0]
                exp_qa = reward_tensor + (gamma * next_qa)

            loss = F.smooth_l1_loss(cur_qa, exp_qa)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            with torch.no_grad():
                loss_value = loss.data.cpu().numpy().item()
                log.debug("Loss value : {}".format(loss_value))
                cacher.save_cacher('loss', loss_value)

    def update(self):
        self.old_qnet.load_state_dict(self.new_qnet.state_dict())
