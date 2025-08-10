import os
import shutil

import torch as th
import torch.nn.functional as F


device = th.device('cuda') if th.cuda.is_available() else th.device('cpu')

FloatTensor = th.FloatTensor if not th.cuda.is_available() else th.cuda.FloatTensor
ByteTensor = th.ByteTensor if not th.cuda.is_available() else th.cuda.ByteTensor


def get_folder(folder, root='trained', makedir=False):
    """ 数据记录（计算图、logs和网络参数）的保存文件路径 """
    folder = os.path.join(root, folder)
    if os.path.exists(folder):
        if makedir:
            shutil.rmtree(folder)
            print('Removing all previous files!')

    log_path = os.path.join(folder, 'logs/')
    if not os.path.exists(log_path) and makedir:
        os.makedirs(log_path)
    graph_path = os.path.join(folder, 'graph/')
    if not os.path.exists(graph_path) and makedir:
        os.makedirs(graph_path)
    model_path = os.path.join(folder, 'model/')
    if not os.path.exists(model_path) and makedir:
        os.makedirs(model_path)

    return {'folder': folder,
            'log_path': log_path,
            'graph_path': graph_path,
            'model_path': model_path}


def soft_update(target, source, tau):
    for target_param, source_param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_((1 - tau) * target_param.data + tau * source_param.data)


def hard_update(target, source):
    for target_param, source_param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(source_param.data)


class LinearSchedule(object):
    def __init__(self, time_steps, final_p, initial_p=1.0):
        self.time_steps = time_steps
        self.final_p = final_p
        self.initial_p = initial_p

    def value(self, t):
        fraction = min(float(t) / self.time_steps, 1.0)
        return self.initial_p + fraction * (self.final_p - self.initial_p)


def onehot_from_logit(logit):
    return F.one_hot(logit.argmax(dim=-1), num_classes=logit.shape[-1])


def sample_gumbel(shape, eps=1e-20):
    u = th.rand(shape)
    return -th.log(-th.log(u + eps) + eps)


def gumbel_softmax(logit, temperature=1.0, hard=False):
    y = logit + sample_gumbel(logit.shape)
    y = F.softmax(y / temperature, dim=-1)
    if hard:
        y_hard = onehot_from_logit(y)
        y = (y_hard - y).detach() + y
    return y


def compute_entropy(q_values, temperature=1.0, cast=False):
    if cast:
        q_values = th.tensor(q_values, dtype=th.float32)
        q_values = q_values.unsqueeze(0)

    # 1. 计算动作概率分布
    action_probs = F.softmax(q_values / temperature, dim=1)  # 归一化概率
    # 2. 计算每个动作的log概率
    log_probs = th.log(action_probs + 1e-10)  # 加小常数避免log(0)
    # 3. 计算熵：batch中每个样本的熵，注意这里对动作维度求和
    entropy = -th.sum(action_probs * log_probs, dim=1)  # [batch_size]
    # 4. 对 batch 求平均
    mean_entropy = entropy.mean()
    return mean_entropy
