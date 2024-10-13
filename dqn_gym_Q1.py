import argparse
import os

import gym
import math
import numpy as np
import time
import torch
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt

from dqn_common import epsilon_by_frame, DqnNetSingleLayer, DqnNetTwoLayers, alpha_sync, DuellingDqn
from lib.experience_buffer import ExperienceBuffer, Experience
import yaml

parser = argparse.ArgumentParser()
# 添加一个可选参数，用于设置强化学习环境的名称。它的默认值为 "CartPole-v1"。
parser.add_argument("-e", "--env", default="CartPole-v1",
                    help="Full name of the environment, e.g. CartPole-v1, LunarLander-v2, etc.")
# 添加一个可选参数，用户可以通过它来指定配置文件的路径。配置文件包含了训练所需的超参数，它的默认值为 "config/dqn.yaml"
parser.add_argument("-c", "--config_file", default="config/dqn.yaml", help="Config file with hyper-parameters")
# 添加一个可选参数，用于设置DQN网络的架构。它的默认值为 "s"，即单隐层网络。
parser.add_argument("-n", "--network", default='s',
                    help="DQN network architecture `single-hidden` for single hidden layer, `two-hidden` for 2 hidden layers and `duelling-dqn` for duelling DQN",
                    choices=['single-hidden', 'two-hidden', 'duelling-dqn'])
# 添加一个可选参数，用户可以通过它来设置随机数生成器的种子。如果不设置，将使用随机种子。
parser.add_argument("-s", "--seed", type=int, help="Manual seed (leave blank for random seed)")
# 解析输入的命令行参数，并将他们存储在 args 变量中
args = parser.parse_args()

# Hyperparameters for the requried environment
# 打开和读取由命令行参数 args.config_file 指定的配置文件。其中包含了训练所需的超参数
hypers = yaml.load(open(args.config_file), Loader=yaml.FullLoader)
# 检查用户选择的环境在超参数文件中是否存在
if args.env not in hypers:
    raise Exception(
        f'Hyper-parameters not found for env {args.env} - please add it to the config file (config/dqn.yaml)')

# 从超参数字典中提取出特定环境的超参数，并将它们存储在 params 变量中
params = hypers[args.env]

# 根据用户指定的环境名称（args.env）来创建 Gym 环境实例
env = gym.make(args.env)

# Set seeds
# 检查是否在命令行参数中设置了随机数种子
if args.seed is not None:
    # 如果设置了随机数种子，将其设置为 PyTorch 和 NumPy 的随机数种子
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
# 检查是否有可用的 GPU
if torch.cuda.is_available():
    device = torch.device("cuda")
    print("Training on GPU")
else:
    device = torch.device("cpu")
    print("Training on CPU")
# 根据命令行参数设置的网络架构选项创建相应的网络实例
if args.network == 'two-hidden':
    net = DqnNetTwoLayers(obs_size=env.observation_space.shape[0],
                          hidden_size=params['hidden_size'], hidden_size2=params['hidden_size2'],
                          n_actions=env.action_space.n).to(device)
    target_net = DqnNetTwoLayers(obs_size=env.observation_space.shape[0],
                                 hidden_size=params['hidden_size'], hidden_size2=params['hidden_size2'],
                                 n_actions=env.action_space.n).to(device)
elif args.network == 'single-hidden':
    net = DqnNetSingleLayer(obs_size=env.observation_space.shape[0],
                            hidden_size=params['hidden_size'],
                            n_actions=env.action_space.n).to(device)
    target_net = DqnNetSingleLayer(obs_size=env.observation_space.shape[0],
                                   hidden_size=params['hidden_size'],
                                   n_actions=env.action_space.n).to(device)
else:
    net = DuellingDqn(obs_size=env.observation_space.shape[0],
                      hidden_size=params['hidden_size'],
                      n_actions=env.action_space.n).to(device)
    target_net = DuellingDqn(obs_size=env.observation_space.shape[0],
                             hidden_size=params['hidden_size'],
                             n_actions=env.action_space.n).to(device)

print(net)

# 创建 SummaryWriter 对象来为 TensorBoard 编写日志文件，方便后续的可视化。
# writer = SummaryWriter(comment="-CartPoleScratch")

# 创建一个 ExperienceBuffer 对象进行经验回放。其大小为 params['replay_size']，在设备 device 上操作
buffer = ExperienceBuffer(int(params['replay_size']), device)
# 为网络 net 创建一个 Adam 优化器。学习率为 params['learning_rate']。
optimizer = optim.Adam(net.parameters(), lr=params['learning_rate'])

frame_idx = 0  # 创建一个变量来跟踪步骤的总数
max_reward = -math.inf  # 初始化最大奖励为负无穷大
all_rewards = []  # 创建一个列表来存储每个周期的奖励
losses = []  # 创建一个列表来存储每个周期的损失
episode_reward = 0  # 初始化周期奖励为 0
r100 = -math.inf  # 初始化过去100个周期的平均奖励为负无穷大
episode_start = time.time()  # 记录当前周期的开始时间
start = time.time()  # 记录整个训练的开始时间
episode_frame = 0  # 初始化每一轮的步数为0
episode_no = 0  # 初始化周期编号为0
visualizer_on = False  # 初始化可视化标志为 False

state, _ = env.reset()  # 重置环境，返回初始的环境状态。对环境的重置通常在一个新游戏轮次开始时进行。它返回的状态会在接下来的训练中使用


def calculate_loss(net, target_net):
    # 从经验回放缓冲区中随机采样 params['batch_size'] 个经验元组
    # 每个样本包括状态、动作、奖励、结束标志以及下一个状态
    states_v, actions_v, rewards_v, dones_v, next_states_v = buffer.sample(params['batch_size'])

    # get the Q value of the state - i.e. Q value for each action possible in that state
    # in CartPole there are 2 actions so this will be tensor of (2, BatchSize)
    # 通过网络的前向传播计算对应于给定状态的所有可能动作的Q值
    Q_s = net.forward(states_v)

    # now we need the state_action_values for the actions that were selected (i.e. the action from the tuple)
    # actions tensor is already {100, 1}, i.e. unsqeezed so we don't need to unsqueeze it again
    # because the Q_s has one row per sample and the actions will be use as indices to choose the value from each row
    # lastly, because the gather will return a column and we need a row, we will squeeze it
    # gather on dim 1 means on rows
    # 从计算得到的Q值中挑选出实际选择的动作对应的Q值，得到的结果将是一个张量，尺寸与批次大小相同
    state_action_values = Q_s.gather(1, actions_v.type(torch.int64).unsqueeze(-1)).squeeze(-1)

    # now we need Q_s_prime_a - i.e. the next state values
    # we get them from the target net
    # because there are 2 actions, we get a tensor of (2, BatchSize)
    # and because it's Sarsa max, we are taking the max
    # .max(1) will find maximum for each row and return a tuple (values, indices) - we need values so get<0>
    # 用目标网络计算下一个状态的Q值，选择其中的最大值，代表了对下一个状态的最优行动的预期回报
    next_state_values = target_net.forward(next_states_v).max(1)[0]

    # calculate expected action values - discounted value of the state + reward
    # 计算预期的Q值，即立即奖励加上折扣后的未来回报。注意如果该步骤是回合的最后一步（由 dones_v 决定），那么未来回报为0
    expected_state_action_values = rewards_v + next_state_values.detach() * params['gamma'] * (1 - dones_v)
    # 计算预期Q值与真实Q值之间的均方误差，作为损失函数
    loss = F.mse_loss(state_action_values, expected_state_action_values)
    # 清除优化器中的梯度信息，以开始新的优化步骤
    optimizer.zero_grad()
    # 对损失进行反向传播，计算每个参数的梯度
    loss.backward()
    # 检查是否设置了梯度裁剪
    if params['clip_gradient']:
        # 如果设置了梯度裁剪，对梯度进行裁剪
        torch.nn.utils.clip_grad_norm_(net.parameters(), 0.1)
    # 根据梯度信息更新网络的参数
    optimizer.step()
    # 返回损失值
    return loss


reward_values_list = []
r100_values_list = []

while True:
    # 记录当前步数
    frame_idx += 1

    # calculate the value of decaying epsilon
    epsilon = epsilon_by_frame(frame_idx, params) # 计算当前步数对应的 epsilon 值
    if np.random.random() < epsilon: # 以 epsilon 的概率进行探索
        # explore
        action = env.action_space.sample()  # 随机选择一个动作
    else:
        # exploit
        state_a = np.array([state], copy=False)  # 将当前状态转为 numpy 数组，为利用策略做准备
        state_v = torch.tensor(state_a).to(device)  # 将当前状态转为 PyTorch tensor，并将其移至硬件设备（CPU 或 GPU）上，为网络输入做准备
        q_vals_v = net(state_v)  # 将当前状态传入网络，计算每个可选动作的 Q值。
        _, act_v = torch.max(q_vals_v, dim=1)  # 找到具有最大 Q 值的动作。torch.max 返回最大值以及对应的索引，此处我们只关心索引（即动作）
        action = act_v.item()  # 获取所选动作的值
        # print(action)

    # take step in the environment
    new_state, reward, terminated, truncated, _ = env.step(action)  # 执行动作，获取新状态、奖励、终止标志和截断标志
    is_done = terminated or truncated  # 如果因为任何原因本轮结束，则认为完成，is_done 设置为 True
    episode_reward += reward  # 累加每一步的奖励，获得总奖励

    # store the transition in the experience replay buffer
    exp = Experience(state, action, reward, is_done, new_state)  # 创建一个经验对象 exp，存储当前交互的信息
    buffer.append(exp)  # 将经验元组添加到经验回放缓冲区
    state = new_state  # 更新当前状态，用于下一步的决策

    # when the episode is done, reset and update progress
    if is_done:  # 判断当前回合是否结束
        done_reward = episode_reward  # 记录本轮的总奖励
        all_rewards.append(episode_reward)  # 如果回合结束，则将本回合的累积奖励记录到列表 all_rewards 中
        episode_no += 1  # 增加回合的计数

        state, _ = env.reset()  # 重置环境，返回初始的环境状态。对环境的重置通常在一个新游戏轮次开始时进行。它返回的状态会在接下来的训练中使用
        if episode_reward > max_reward:  # 如果本轮的总奖励大于历史最大奖励
            max_reward = episode_reward  # 更新最大奖励

        if len(all_rewards) > 10 and len(losses) > 10:   # 当已有足够多的回合和损失时，开始计算某些统计指标
            r100 = np.mean(all_rewards[-100:])  # 计算过去100个周期的平均奖励
            l100 = np.mean(losses[-100:])  # 计算过去100个周期的平均损失
            fps = (frame_idx - episode_frame) / (time.time() - episode_start)  # 计算每秒的步数
            print(
                f"Frame: {frame_idx}: Episode: {episode_no}, R100: {r100: .2f}, MaxR: {max_reward: .2f}, R: {episode_reward: .2f}, FPS: {fps: .1f}, L100: {l100: .2f}, Epsilon: {epsilon: .4f}")
            r100_values_list.append(r100)
            reward_values_list.append(episode_reward)


            # visualize the training when reachedd 95% of the target R100
            if not visualizer_on and r100 > 0.95 * params['stopping_reward']:  # 如果当前平均奖励达到目标的95%
                env = gym.make(args.env, render_mode='human')  # 创建一个新的环境实例，用于可视化
                env.reset()  # 重置环境
                env.render()  # 渲染环境
                visualizer_on = True  # 设置可视化标志为 True

        episode_reward = 0  # 重置本轮的总奖励
        episode_frame = frame_idx  # 重置本轮的步数
        episode_start = time.time()   # 重置本轮的开始时间

    if len(buffer) < params['replay_size_start']:  # 如果经验回放缓冲区中的经验元组数量小于 params['replay_size_start']
        continue  # 继续下一步


    # do the learning
    loss = calculate_loss(net, target_net)  # 调用前面定义的 calculate_loss 函数计算损失
    losses.append(loss.item())  # 将此次计算的损失值添加到记录中

    if params['alpha_sync']:  # 检查是否使用 alpha 同步更新目标网络
        alpha_sync(net, target_net, alpha=1 - params['tau'])  # 如果设置了 alpha 同步，将当前网络的权重按照一定比例同步到目标网络
    elif frame_idx % params['target_net_sync'] == 0:  # 如果没有设置 alpha 同步，但到了设定的同步间隔
        target_net.load_state_dict(net.state_dict())  # 直接将当前网络的参数复制到目标网络中

    if r100 > params['stopping_reward']:  # 检查最近 100 个回合的平均奖励是否达到了设定的停止训练门槛
        print("Finished training")  # 如果训练达到了目标，输出完成训练的信息

        name = f"{args.env}_{args.network}_nn_DQN_act_net_%+.3f_%d.dat" % (r100, frame_idx) # 生成模型文件的名称
        if not os.path.exists(params['save_path']):  # 如果保存路径不存在
            os.makedirs(params['save_path'])  # 创建保存路径
        torch.save(net.state_dict(), os.path.join(params['save_path'], name))  # 保存网络

        break

    if frame_idx > params['max_frames']:  # 如果训练步数超过了最大步数
        print(f"Ran out of time at {time.time() - start}") # 输出训练结束的信息
        break


def plot_rewards(r100_values, reward_values, environment, network):
    plt.figure(figsize=(10, 5))
    plt.plot(r100_values, label='R100')
    plt.plot(reward_values, label='Episode Reward')
    plt.title(f'{environment}-{network} Rewards vs Episode Number')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.legend()
    plt.grid(True)

    images_dir = 'Q1_Images'
    if not os.path.exists(images_dir):
        os.makedirs(images_dir)
    filename = os.path.join(images_dir, f'{environment}-{network}.png')
    plt.savefig(filename)
    plt.show()


plot_rewards(r100_values_list, reward_values_list, args.env, args.network)
