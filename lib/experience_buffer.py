import collections
from collections import namedtuple
import numpy as np
import torch

Experience = collections.namedtuple(
    'Experience',  # 经验元组命名
    field_names=['state', 'action', 'reward', 'done', 'new_state'])  # 字段名
ExperienceImageHistory = collections.namedtuple(
    'ExperienceImageHistory',  # 经验元组命名
    field_names=['history', 'state', 'action', 'reward', 'done', 'new_state'])  # 字段名


class ExperienceBuffer():
    # 定义了一个名为 ExperienceBuffer 的类。这是一个用于存储经验元组的环形缓冲区
    # 初始化函数，接收两个参数，缓冲区的容量 capacity 和设备 device。
    def __init__(self, capacity, device):
        self.buffer = collections.deque(maxlen=capacity)  # 使用python的双端队列来创建一个固定大小的环形缓冲区
        self.device = device  # 存储设备信息
        self.size = 0  # 记录缓冲区中的元素个数
        self.capacity = capacity  # 记录缓冲区的容量

    def __len__(self):
        return len(self.buffer)  # 返回缓冲区中的元素个数

    def append(self, experience):
        self.buffer.append(experience)  # 将经验元组添加到缓冲区中
        if self.size < self.capacity:  # 如果缓冲区未满
            self.size += 1  # 更新缓冲区中的元素个数

    def sample(self, batch_size):
        indices = np.random.choice(self.size, batch_size, replace=False)  # 从缓冲区中随机采样 batch_size 个元素
        states, actions, rewards, dones, next_states = zip(*[self.buffer[idx] for idx in indices])
        # 将采样的经验元组拆分成状态、动作、奖励、终止标志和下一个状态
        states_input = states if isinstance(states, list) else np.array(states)  # 将状态转换为 numpy 数组
        next_states_input = next_states if isinstance(next_states, list) else np.array(
            next_states)  # 将下一个状态转换为 numpy 数组

        # 将相应的组件转换成 PyTorch 的 Tensor 数据类型，并将数据移动到指定的设备上
        return torch.tensor(states_input, dtype=torch.float).to(self.device), \
            torch.tensor(np.array(actions)).to(self.device), \
            torch.tensor(np.array(rewards, dtype=np.float32)).to(self.device), \
            torch.tensor(np.array(dones, dtype=np.uint8)).to(self.device), \
            torch.tensor(next_states_input, dtype=torch.float).to(self.device)


class ExperienceBufferWithHistory(ExperienceBuffer):
    # 定义了一个名为 ExperienceBufferWithHistory 的类，继承自 ExperienceBuffer 类
    def __init__(self, capacity):
        super().__init__(capacity)  # 初始化函数，接收一个参数：缓冲区的容量 capacity

    # 定义了一个从缓冲区中随机抽样的方法。返回一个包含 batch_size 大小的经验样本集，
    # 和基类的区别在于，此方法没有将样本集转换为 Torch 的 Tensor 数据类型。
    def sample(self, batch_size):
        # 从缓冲区中随机采样 batch_size 个元素
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        # 将采样的经验元组拆分成状态、动作、奖励、终止标志和下一个状态
        states, actions, rewards, dones, next_states = zip(*[self.buffer[idx] for idx in indices])
        # states and next_states are already np arrays - they were pre-processed through RNN
        # 返回这批抽样出的经验，它们分别是状态、动作、奖励、是否结束、新状态。
        # 需要注意的是这里并没有将数据转换为 Torch 的 Tensor 数据类型
        return states, \
            np.array(actions), \
            np.array(rewards, dtype=np.float32), \
            np.array(dones, dtype=np.uint8), \
            next_states


class ExperienceBufferImageHistory(ExperienceBuffer):
    # 定义了一个名为 ExperienceBufferImageHistory 的类，继承自 ExperienceBuffer 类
    def __init__(self, capacity):
        super().__init__(capacity)  # 初始化函数，接收一个参数：缓冲区的容量 capacity

    # 用于从缓冲区中随机抽取指定数量（batch_size）的元素（即“经验”）
    def sample(self, batch_size):
        # 从缓冲区中随机采样 batch_size 个元素，这个函数随机生成 batch_size 个索引，根据索引从缓冲区中取出对应的经验
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        histories, states, actions, rewards, dones, next_states = zip(*[self.buffer[idx] for idx in indices])
        # states and next_states are already np arrays - they were pre-processed through RNN
        # 返回这批抽样出的经验，它们分别是状态、动作、奖励、是否结束、新状态。
        return np.array(histories), \
            np.array(states), \
            np.array(actions), \
            np.array(rewards, dtype=np.float32), \
            np.array(dones, dtype=np.uint8), \
            np.array(next_states)


# 这是一个用于将 histories 填充到指定大小 to_size 的函数
def pad_with_zeros(histories, to_size, pad_value=-1000):
    if len(histories) < to_size:  # 如果 histories 的长度小于 to_size，则需要进行填充
        # 创建一个长度为 len(histories[0])，所有元素都是 pad_value 的数组，并转换为列表
        padding = (np.zeros(len(histories[0])) + pad_value).tolist()
        # 将 padding 插入到 histories 的头部，使得 histories 的长度等于 to_size
        for _ in range(to_size - len(histories)):
            histories.insert(0, padding)

    return histories  # 在填充完历史记录后，返回填充后的历史记录
