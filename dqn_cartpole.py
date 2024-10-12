import argparse

import gym
import math
import numpy as np
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from lib.experience_buffer import ExperienceBuffer, Experience
import os


# Define the network structure - in this case 2 hidden layers (CartPole can be solved faster with a single hidden layer)
class DqnNet(nn.Module):
    # 初始化函数。obs_size 是输入的大小（观测的维度），hidden_size 是隐藏层的大小，n_actions 是输出的大小（动作的数量）。
    def __init__(self, obs_size, hidden_size, n_actions):
        super(DqnNet, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_size, hidden_size),  # 第一个全连接层，从 obs_size 维度连接到 hidden_size 维度。
            nn.ReLU(),  # 使用 ReLU 激活函数。
            nn.Linear(hidden_size, n_actions)  # 最后的全连接层，从 hidden_size 维度连接到 n_actions 维度，用于预测每个动作的 Q 值。
        )

    def forward(self, x):
        return self.net(x.float())


def epsilon_by_frame(frame_idx):
    return EPSILON_FINAL + (EPSILON_START - EPSILON_FINAL) * math.exp(-1.0 * frame_idx / EPSILON_DECAY)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

GAMMA = 0.99
BATCH_SIZE = 128
REPLAY_SIZE = 10000
REPLAY_START_SIZE = 10000

HIDDEN_SIZE = 128

EPSILON_DECAY = 5000
EPSILON_FINAL = 0.01
EPSILON_START = 1.00

LEARNING_RATE = 1e-3

TARGET_NET_SYNC = 1e3

STOP_REWARD = 195

ENV = "CartPole-v0"
SAVED_MODELS_PATH = 'saved_models'

env = gym.make(ENV)
# env.render()
# 创建决策网络和目标网络，这两个网络的架构和参数都一样
net = DqnNet(obs_size=env.observation_space.shape[0], hidden_size=HIDDEN_SIZE, n_actions=env.action_space.n).to(device)
target_net = DqnNet(obs_size=env.observation_space.shape[0], hidden_size=HIDDEN_SIZE, n_actions=env.action_space.n).to(
    device)
# 打印出决策网络的架构信息
print(net)

# 创建 SummaryWriter 对象，方便后续在 TensorBoard 中可视化数据。
# writer = SummaryWriter(comment="-CartPoleScratch")

# 创建 ExperienceBuffer 对象，用于存储经验
buffer = ExperienceBuffer(REPLAY_SIZE, device)
# 创建优化器 Adam，用于更新网络参数
optimizer = optim.Adam(net.parameters(), lr=LEARNING_RATE)

# 开始训练时，已运行的游戏帧的总数
frame_idx = 0
# 到目前为止获取的最大的奖励
max_reward = -10000
# 存储每次游戏的总奖励
all_rewards = []
# 存储每次游戏的损失
losses = []
# 当前游戏的总奖励
episode_reward = 0
# 近100次游戏的平均奖励
r100 = 0
# 当前游戏开始的时间
episode_start = time.time()
# 训练开始的时间
start = time.time()
# 当前游戏已运行的帧数
episode_frame = 0
# 重置游戏环境，获得初始状态
state, _ = env.reset()


def calculate_loss(net, target_net):
    # 从 ExperienceBuffer 中随机采样，每个样本包含了状态，动作，奖励，是否结束以及下一个状态
    states_v, actions_v, rewards_v, dones_v, next_states_v = buffer.sample(BATCH_SIZE)

    # get the Q value of the state - i.e. Q value for each action possible in that state
    # in CartPole there are 2 actions so this will be tensor of (2, BatchSize)
    # 使用网络 net 计算每个状态对应的所有动作的Q值
    Q_s = net.forward(states_v)

    # now we need the state_action_values for the actions that were selected (i.e. the action from the tuple)
    # actions tensor is already {100, 1}, i.e. unsqeezed so we don't need to unsqueeze it again
    # 表示actions张量已经具有形状 {100,1}，也就是说它已经被扩张了额外的维度，因此无需再次进行扩张
    # because the Q_s has one row per sample and the actions will be used as indices to choose the value from each row
    # 是说张量Q_s拥有与采样批次大小一样多的行，每行对应一个样本的所有可能动作的Q值。
    # 而actions则被用作索引，用于从每行中选择实际采取动作的Q值
    # lastly, because the gather will return a column and we need a row, we will squeeze it
    # gather on dim 1 means on rows
    # 是说gather操作在 dim=1 的维度上进行，即在行上进行。这实际上将从Q_s的每一行中提取出我们需要的Q值。
    # 通过所选择的动作获取在当前状态下执行该动作的预期Q值
    state_action_values = Q_s.gather(1, actions_v.type(torch.int64).unsqueeze(-1)).squeeze(-1)

    # now we need Q_s_prime_a - i.e. the next state values
    # we get them from the target net
    # because there are 2 actions, we get a tensor of (2, BatchSize)
    # and because it's Sarsa max, we are taking the max
    # .max(1) will find maximum for each row and return a tuple (values, indices) - we need values so get<0>
    # 使用目标网络 target_net 计算下一个状态的最大Q值
    next_state_values = target_net.forward(next_states_v).max(1)[0]

    # calculate expected action values - discounted value of the state + reward
    # 计算期望的动作状态值，就是采取动作后的奖励加上折扣因子乘以下一个状态的最大Q值
    expected_state_action_values = rewards_v + next_state_values.detach() * GAMMA * (1 - dones_v)

    # 计算损失
    loss = F.mse_loss(state_action_values, expected_state_action_values)
    # 梯度归零
    optimizer.zero_grad()
    # 使用自动微分对损失进行反向传播
    loss.backward()
    # 对梯度进行裁剪（这行代码被注释掉了）
    # torch.nn.utils.clip_grad_norm_(net.parameters(), 0.1)
    # 根据目标函数的梯度使用优化器更新网络的参数
    optimizer.step()

    return loss


while True:
    # 训练每进行一步，frame_idx+1
    frame_idx += 1

    # 确定在当前状态下执行的动作。一部分情况下，智能体将探索新的动作，一部分情况下根据当前估计的Q值选择最优动作
    # calculate the value of decaying epsilon
    # 计算变化的epsilon值，控制探索和利用的平衡
    epsilon = epsilon_by_frame(frame_idx)
    # 随机探索策略
    if np.random.random() < epsilon:
        # 如果 true，执行随机动作（探索）
        action = env.action_space.sample()

    # 利用策略
    else:
        # 如果 false，使用当前Q网络计算贪婪动作（利用）
        # 将状态转化为网络可以接受的输入形式
        state_a = np.array([state], copy=False)
        state_v = torch.tensor(state_a).to(device)
        # 计算每个动作的Q值
        q_vals_v = net(state_v)
        # 选择具有最高Q值的动作
        _, act_v = torch.max(q_vals_v, dim=1)
        # 提取该动作的整数值
        action = int(act_v.item())
        # print(action)

    # take step in the environment
    # 执行选择的动作并从环境中获取反馈，包括新的状态，奖励，以及是否到达终止状态。
    new_state, reward, terminated, truncated, _ = env.step(action)
    is_done = terminated or truncated
    episode_reward += reward

    # store the transition in the experience replay buffer
    # 将当前的状态转换、动作、奖励、下一状态，以及是否终止信息作为一个体验存储到重放缓冲池中。
    exp = Experience(state, action, reward, is_done, new_state)
    buffer.append(exp)
    state = new_state

    # when the episode is done, reset and update progress
    # 如果当前游戏结束，那么将重置环境，并且把本次游戏的总奖励添加到总奖励列表中。
    if is_done:
        done_reward = episode_reward
        all_rewards.append(episode_reward)
        state, _ = env.reset()
        # 如果本次游戏的奖励超过目前记录的最大奖励，那么就更新最大奖励值。
        if episode_reward > max_reward:
            max_reward = episode_reward
        # 当我们积累了足够多的奖励和损失记录后（超过101个）
        if len(all_rewards) > 101 and len(losses) > 101:
            r100 = np.mean(all_rewards[-100:]) # 计算最近100次游戏的平均奖励
            l100 = np.mean(losses[-100:]) # 计算最近100次游戏的平均损失
            fps = (frame_idx - episode_frame) / (time.time() - episode_start) # 计算帧率，即每秒游戏的步数
            print(
                f"Frame: {frame_idx}: R100: {r100: .2f}, "
                f"MaxR: {max_reward: .2f}, R: {episode_reward: .2f}, "
                f"FPS: {fps: .1f}, L100: {l100: .2f}, Epsilon: {epsilon: .4f}")

        episode_reward = 0  # 游戏结束后，当前游戏的奖励要重置为0
        episode_frame = frame_idx  # 又由于新游戏开始，所以要记录当前的帧数作为新的游戏开始时的帧数
        episode_start = time.time()  # 记录新游戏开始的时间

    # 如果缓冲区有足够经验再进行学习，否则继续采集经验
    if len(buffer) < REPLAY_START_SIZE:
        continue

    # do the learning
    # 计算损失、优化网络
    loss = calculate_loss(net, target_net)  # 调用 calculate_loss 函数来计算预测的Q值与目标Q值之间的差距。
    losses.append(loss.item())  # 将损失值添加到损失列表中

    # 每隔 TARGET_NET_SYNC 步，就会更新目标网络的参数
    if frame_idx % TARGET_NET_SYNC == 0:
        target_net.load_state_dict(net.state_dict())  # 用 net（预测网络）当前的状态字典来更新 target_net（目标网络）

    # 如果最近100个回合的平均奖励超过195（即 r100 > 195），则表示训练达到了预期效果，停止训练
    if r100 > 195:
        print("Finished training")
        name = f"{ENV}_{HIDDEN_SIZE}_hidden_size_DQN_act_net_%+.3f_%d.dat" % (r100, frame_idx)
        if not os.path.exists(SAVED_MODELS_PATH):
            os.makedirs(SAVED_MODELS_PATH)
        torch.save(net.state_dict(), os.path.join(SAVED_MODELS_PATH, name))

        break

    # 训练太长时间也停止训练
    if frame_idx > 100000:
        print(f"Ran out of time at {time.time() - start}")
        break
