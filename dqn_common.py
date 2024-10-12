import math

import torch.nn as nn


# Define the network structure - in this case 2 hidden layers
# (CartPole can be solved faster with a single hidden layer)
# 定义一个神经网络类，继承自pytorch的nn.Module模块
class DqnNetTwoLayers(nn.Module):

    # 初始化函数，obs_size是输入的大小（观测的维度），hidden_size和hidden_size2是两个隐含层的大小，
    # n_actions是输出的大小（动作的数量）
    def __init__(self, obs_size, hidden_size, hidden_size2, n_actions):
        # 调用父类的初始化函数
        super(DqnNetTwoLayers, self).__init__()

        # 定义神经网络的结构。这里使用Sequential容器，表示神经网络层的线性堆叠。包含两个全连接层（Linear)
        # 每一层后面跟着一个ReLU激活函数，最后是一个用于产生Q值预测的全连接层
        self.net = nn.Sequential(
            nn.Linear(obs_size, hidden_size),  # 第一个全连接层，从obs_size维度连接到hidden_size维度
            nn.ReLU(),  # 使用ReLU激活函数
            nn.Linear(hidden_size, hidden_size2),  # 第二个全连接层，从hidden_size维度连接到hidden_size2维度
            nn.ReLU(),  # 使用ReLU激活函数
            nn.Linear(hidden_size2, n_actions)  # 最后的全连接层，从hidden_size2维度连接到n_action维度，预测每个动作的Q值
        )

    # 定义前向传播过程，即如何根据输入x计算出输出
    def forward(self, x):
        return self.net(x.float())  # 将输入转换为浮点类型，并使用self.net计算输出


# 定义一个神经网络类，继承自pytorch的nn.Module模块
class DqnNetSingleLayer(nn.Module):

    # 初始化函数，obs_size是输入的大小（观测的维度），hidden_size是隐藏层的大小，n_actions是输出的大小（动作的数量）
    def __init__(self, obs_size, hidden_size, n_actions):
        # 调用父类的初始化函数
        super(DqnNetSingleLayer, self).__init__()

        # 定义神经网络的结构。这里使用Sequential容器，表示神经网络层的线性堆叠。包含一个全连接层（Linear)，
        # 每一层后面跟着一个ReLU激活函数，最后是一个用于产生Q值预测的全连接层
        self.net = nn.Sequential(
            nn.Linear(obs_size, hidden_size),  # 第一个全连接层，从obs_size维度连接到hidden_size维度
            nn.ReLU(),  # 使用ReLU激活函数
            nn.Linear(hidden_size, n_actions)  # 最后的全连接层，从hidden_size维度连接到n_action维度，预测每个动作的Q值
        )

    # 定义前向传播过程，即如何根据输入x计算出输出
    def forward(self, x):
        return self.net(x.float())  # 将输入转换为浮点类型，并使用self.net计算输出


# 定义自我对抗神经网络类，继承自pytorch的nn.Module模块
class DuellingDqn(nn.Module):
    # 初始化函数，obs_size是输入的大小（观测的维度），hidden_size是隐藏层的大小，n_actions是输出的大小（动作的数量）
    def __init__(self, obs_size, hidden_size, n_actions):
        # 调用父类的初始化函数
        super(DuellingDqn, self).__init__()
        # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        # we have 2 nets now - one for values and one for advantage (i.e. the difference each action causes)
        # with 2 layers it doesn't converge!!!!!!!!!!!!!!!!!
        # 定义状态值网络，这是一个线性层和ReLU激活函数的序列，最后是一个输出单一状态值的线性层
        self.value_net = nn.Sequential(
            nn.Linear(obs_size, hidden_size),
            # nn.ReLU(),
            # nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
            # careful - the output here is just a single state value
            nn.Linear(hidden_size, 1)
        )

        # 定义优势网络，这是一个线性层和ReLU激活函数的序列，最后是一个输出每个动作优势的线性层
        self.advantage_net = nn.Sequential(
            nn.Linear(obs_size, hidden_size),
            # nn.ReLU(),
            # nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, n_actions)
        )

    # 定义前向传播过程，即如何根据输入x计算出输出
    def forward(self, x):
        # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        # Q = value + advantage - advantage.mean()
        # 计算状态值
        value = self.value_net(x.float())
        # 计算每个动作的优势
        advantage = self.advantage_net(x.float())
        # 计算Q值：状态值 + 动作优势 - 动作优势的平均值
        return value + advantage - advantage.mean()


# 该函数用于计算在给定时间步（frame_idx）应使用的ε值
def epsilon_by_frame(frame_idx, params):
    # ε值由一个初始值线性下降到一个最终值，下降速度由params字典中的 'epsilon_decay' 参数控制
    return params['epsilon_final'] + (params['epsilon_start'] - params['epsilon_final']) * math.exp(
        -1.0 * frame_idx / params['epsilon_decay'])


# 该函数用于将主网络（net）的参数通过权重α同步到目标网络（tgt_net）
def alpha_sync(net, tgt_net, alpha):
    # 检查alpha是否为float类型，且在0到1之间
    assert isinstance(alpha, float)
    assert 0.0 < alpha <= 1.0
    # 获取主网络和目标网络的参数字典
    state = net.state_dict()
    tgt_state = tgt_net.state_dict()
    # 遍历主网络的每个参数，将它同步到目标网络的对应参数上
    for k, v in state.items():
        tgt_state[k] = tgt_state[k] * alpha + (1 - alpha) * v
    # 将更新后的参数重新加载到目标网络
    tgt_net.load_state_dict(tgt_state)
