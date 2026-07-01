"""
基于 DQN 的智能仓储机器人路径规划与货架拣选策略
======================================================
一键运行: python warehouse_grid_dqn.py

依赖: numpy, matplotlib, torch
安装: pip install numpy matplotlib torch

作者: [学生姓名]
日期: 2026-05
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from collections import deque
import random
import os

# 设置中文字体（如果系统支持）
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


# ============================================================================
# 第一部分：仓库网格环境 (Warehouse Grid Environment)
# ============================================================================

class WarehouseGridEnv:
    """
    智能仓储网格环境

    网格元素编码:
        0 - 空地 (可通行)
        1 - 障碍物 (不可通行)
        2 - 起点
        3 - 目标货架

    状态空间: 机器人当前坐标 (x, y)
    动作空间: 0=上, 1=下, 2=左, 3=右
    """

    def __init__(self, grid_size=10, obstacle_ratio=0.15, num_goals=1,
                 fixed_obstacles=None, start_pos=None, goal_positions=None):
        """
        初始化仓库环境

        参数:
            grid_size: 网格大小 (grid_size × grid_size)
            obstacle_ratio: 随机障碍物比例 (当 fixed_obstacles=None 时使用)
            num_goals: 目标货架数量
            fixed_obstacles: 固定的障碍物坐标列表，若为 None 则随机生成
            start_pos: 起点坐标 (x,y)，若为 None 则随机生成
            goal_positions: 目标货架坐标列表，若为 None 则随机生成
        """
        self.grid_size = grid_size
        self.obstacle_ratio = obstacle_ratio
        self.num_goals = num_goals

        # 动作映射: 0=上, 1=下, 2=左, 3=右
        self.action_space = 4
        self.action_map = {
            0: (0, -1),   # 上 (y-1)
            1: (0, 1),    # 下 (y+1)
            2: (-1, 0),   # 左 (x-1)
            3: (1, 0),    # 右 (x+1)
        }
        self.action_names = ['上', '下', '左', '右']

        # 奖励设置
        self.step_penalty = -1.0          # 每步负奖励
        self.goal_reward = 50.0           # 到达目标正奖励
        self.collision_penalty = -20.0    # 碰撞负奖励
        self.revisit_penalty = -2.0       # 重复访问负奖励（扩展）

        # 环境元素
        self.fixed_obstacles = fixed_obstacles
        self.fixed_start = start_pos
        self.fixed_goals = goal_positions

        # 内部状态
        self.grid = None
        self.robot_pos = None
        self.goals = None
        self.visited = None
        self.steps = 0
        self.max_steps = grid_size * grid_size * 3  # 最大步数限制
        self.done = False
        self.success = False
        self.path_history = []

        self.reset()

    def reset(self):
        """重置环境，返回初始状态"""
        # 初始化网格 (全空地)
        self.grid = np.zeros((self.grid_size, self.grid_size), dtype=int)

        # 放置障碍物
        if self.fixed_obstacles is not None:
            for ox, oy in self.fixed_obstacles:
                self.grid[ox, oy] = 1
        else:
            num_obstacles = int(self.grid_size * self.grid_size * self.obstacle_ratio)
            obstacles = set()
            while len(obstacles) < num_obstacles:
                ox = np.random.randint(0, self.grid_size)
                oy = np.random.randint(0, self.grid_size)
                obstacles.add((ox, oy))
            for ox, oy in obstacles:
                self.grid[ox, oy] = 1

        # 放置起点
        if self.fixed_start is not None:
            self.robot_pos = list(self.fixed_start)
        else:
            while True:
                sx = np.random.randint(0, self.grid_size)
                sy = np.random.randint(0, self.grid_size)
                if self.grid[sx, sy] == 0:
                    self.robot_pos = [sx, sy]
                    break
        self.start_pos = tuple(self.robot_pos)

        # 放置目标货架
        if self.fixed_goals is not None:
            self.goals = [list(g) for g in self.fixed_goals]
        else:
            self.goals = []
            for _ in range(self.num_goals):
                while True:
                    gx = np.random.randint(0, self.grid_size)
                    gy = np.random.randint(0, self.grid_size)
                    if (self.grid[gx, gy] == 0 and
                        [gx, gy] != self.robot_pos and
                        [gx, gy] not in self.goals):
                        self.goals.append([gx, gy])
                        break

        # 在网格上标记起点和目标（用于渲染）
        self._update_grid_markers()

        # 重置内部状态
        self.visited = set()
        self.visited.add(tuple(self.robot_pos))
        self.steps = 0
        self.done = False
        self.success = False
        self.path_history = [tuple(self.robot_pos)]

        return self._get_state()

    def _update_grid_markers(self):
        """更新网格上的起点和目标标记"""
        # 清除之前的标记（保留障碍物）
        self.grid[self.grid == 2] = 0
        self.grid[self.grid == 3] = 0
        # 标记起点
        self.grid[self.start_pos[0], self.start_pos[1]] = 2
        # 标记目标
        for g in self.goals:
            self.grid[g[0], g[1]] = 3

    def _get_state(self):
        """
        获取当前状态表示

        返回归一化的坐标, 形状为 (2,)，值域 [0, 1]
        """
        return np.array([
            self.robot_pos[0] / max(self.grid_size - 1, 1),
            self.robot_pos[1] / max(self.grid_size - 1, 1),
        ], dtype=np.float32)

    def step(self, action):
        """
        执行动作，返回 (next_state, reward, done, info)

        参数:
            action: 0=上, 1=下, 2=左, 3=右

        返回:
            next_state: 下一个状态
            reward: 奖励值
            done: 是否结束
            info: 额外信息字典
        """
        if self.done:
            return self._get_state(), 0.0, True, {'success': self.success}

        dx, dy = self.action_map[action]
        new_x = self.robot_pos[0] + dx
        new_y = self.robot_pos[1] + dy
        self.steps += 1

        reward = self.step_penalty  # 基础步数惩罚

        # 检查是否出界
        if not (0 <= new_x < self.grid_size and 0 <= new_y < self.grid_size):
            reward += self.collision_penalty
            self.done = True
            self.success = False
            info = {'success': False, 'reason': 'out_of_bounds'}
            return self._get_state(), reward, True, info

        # 检查是否撞到障碍物
        if self.grid[new_x, new_y] == 1:
            reward += self.collision_penalty
            self.done = True
            self.success = False
            info = {'success': False, 'reason': 'collision'}
            return self._get_state(), reward, True, info

        # 移动到新位置
        self.robot_pos = [new_x, new_y]
        self.path_history.append(tuple(self.robot_pos))

        # 检查重复访问
        pos_tuple = tuple(self.robot_pos)
        if pos_tuple in self.visited:
            reward += self.revisit_penalty
        self.visited.add(pos_tuple)

        # 检查是否到达目标
        if self.robot_pos in self.goals:
            reward += self.goal_reward
            # 额外奖励：步数越少奖励越多
            step_bonus = max(0, self.max_steps // 4 - self.steps) * 0.5
            reward += step_bonus
            self.done = True
            self.success = True
            info = {'success': True, 'reason': 'reached_goal', 'steps': self.steps}
            return self._get_state(), reward, True, info

        # 检查是否超过最大步数
        if self.steps >= self.max_steps:
            self.done = True
            self.success = False
            info = {'success': False, 'reason': 'timeout', 'steps': self.steps}
            return self._get_state(), reward, True, info

        info = {'success': False, 'reason': 'moving'}
        return self._get_state(), reward, False, info

    def render_text(self):
        """文本方式渲染网格"""
        symbols = {0: '.', 1: '#', 2: 'S', 3: 'G'}
        grid_copy = self.grid.copy()
        rx, ry = self.robot_pos

        print('=' * (self.grid_size * 2 + 2))
        for y in range(self.grid_size):
            row = '|'
            for x in range(self.grid_size):
                if [x, y] == self.robot_pos:
                    row += 'R '  # Robot
                elif grid_copy[x, y] == 3:
                    row += 'G '
                elif grid_copy[x, y] == 2:
                    row += 'S '
                elif grid_copy[x, y] == 1:
                    row += '# '
                else:
                    row += '. '
            row += '|'
            print(row)
        print('=' * (self.grid_size * 2 + 2))
        print(f"位置: ({rx},{ry})  |  步数: {self.steps}")
        print(f"已访问: {len(self.visited)} 格  |  剩余目标: {len([g for g in self.goals if g != self.robot_pos])}")

    def render_matplotlib(self, ax=None, show_path=False, title="仓库网格环境"):
        """
        Matplotlib 可视化渲染

        参数:
            ax: matplotlib axes 对象
            show_path: 是否显示历史路径
            title: 图表标题
        """
        if ax is None:
            _, ax = plt.subplots(figsize=(7, 7))

        # 绘制网格背景
        grid_display = np.zeros((self.grid_size, self.grid_size))
        for x in range(self.grid_size):
            for y in range(self.grid_size):
                if self.grid[x, y] == 1:
                    grid_display[x, y] = -1  # 障碍物

        # 着色
        cmap = plt.cm.colors.ListedColormap(['white', 'gray', 'lightgreen', 'gold'])
        bounds = [-1.5, -0.5, 0.5, 1.5, 2.5]
        norm = plt.cm.colors.BoundaryNorm(bounds, cmap.N)

        ax.imshow(grid_display.T, cmap=cmap, norm=norm, origin='lower',
                  extent=[-0.5, self.grid_size - 0.5, -0.5, self.grid_size - 0.5])

        # 绘制网格线
        for i in range(self.grid_size + 1):
            ax.axhline(i - 0.5, color='black', linewidth=0.5)
            ax.axvline(i - 0.5, color='black', linewidth=0.5)

        # 标记起点
        ax.add_patch(patches.Rectangle(
            (self.start_pos[0] - 0.4, self.start_pos[1] - 0.4), 0.8, 0.8,
            linewidth=2, edgecolor='blue', facecolor='lightblue', label='起点'
        ))
        ax.text(self.start_pos[0], self.start_pos[1], 'S', ha='center', va='center',
                fontweight='bold', fontsize=12, color='blue')

        # 标记目标
        for i, g in enumerate(self.goals):
            ax.add_patch(patches.Rectangle(
                (g[0] - 0.4, g[1] - 0.4), 0.8, 0.8,
                linewidth=2, edgecolor='green', facecolor='lightgreen', label='目标' if i == 0 else ''
            ))
            ax.text(g[0], g[1], 'G', ha='center', va='center',
                    fontweight='bold', fontsize=12, color='green')

        # 标记机器人当前位置
        rx, ry = self.robot_pos
        ax.plot(rx, ry, 'o', color='red', markersize=18, markeredgecolor='darkred',
                markeredgewidth=2, zorder=5, label='机器人')
        ax.text(rx, ry, 'R', ha='center', va='center', fontweight='bold',
                fontsize=8, color='white', zorder=6)

        # 显示历史路径
        if show_path and len(self.path_history) > 1:
            px = [p[0] for p in self.path_history]
            py = [p[1] for p in self.path_history]
            ax.plot(px, py, '-', color='orange', linewidth=2, alpha=0.7, label='路径')
            # 起点箭头
            for i in range(len(px) - 1):
                ax.annotate('', xy=(px[i + 1], py[i + 1]), xytext=(px[i], py[i]),
                            arrowprops=dict(arrowstyle='->', color='orange',
                                          lw=1.5, alpha=0.5))

        ax.set_xlim(-0.5, self.grid_size - 0.5)
        ax.set_ylim(-0.5, self.grid_size - 0.5)
        ax.set_xticks(range(self.grid_size))
        ax.set_yticks(range(self.grid_size))
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_title(title)
        ax.legend(loc='upper right', fontsize=8)
        ax.set_aspect('equal')

        return ax

    def get_manhattan_distance_to_goal(self):
        """计算到最近目标的曼哈顿距离（用于评估）"""
        if not self.goals:
            return 0
        return min(abs(self.robot_pos[0] - g[0]) + abs(self.robot_pos[1] - g[1])
                   for g in self.goals)


# ============================================================================
# 第二部分：DQN 网络结构
# ============================================================================

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


class DQN(nn.Module):
    """
    标准 DQN 网络 (MLP)

    输入: 状态向量 (2,) — 归一化的 (x, y) 坐标
    输出: 各动作的 Q 值 (4,) — [上, 下, 左, 右]
    """

    def __init__(self, state_dim, action_dim, hidden_dims=(128, 128)):
        """
        参数:
            state_dim: 状态维度
            action_dim: 动作维度
            hidden_dims: 隐藏层维度元组
        """
        super(DQN, self).__init__()

        layers = []
        input_dim = state_dim
        for h_dim in hidden_dims:
            layers.append(nn.Linear(input_dim, h_dim))
            layers.append(nn.ReLU(inplace=True))
            input_dim = h_dim
        layers.append(nn.Linear(input_dim, action_dim))

        self.network = nn.Sequential(*layers)

        # 初始化权重
        self._init_weights()

    def _init_weights(self):
        """使用 Xavier 初始化权重"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, state):
        """前向传播，返回 Q(s, a)"""
        return self.network(state)


class DuelingDQN(nn.Module):
    """
    Dueling DQN 网络结构（加分项）

    将 Q 值分解为状态价值 V(s) 和优势函数 A(s, a):
        Q(s, a) = V(s) + A(s, a) - mean(A(s, a))

    这种分解使得网络可以分别学习状态的价值和各个动作的相对优势，
    在动作之间价值差异不大的情况下，学习效率更高。
    """

    def __init__(self, state_dim, action_dim, hidden_dims=(128, 128)):
        super(DuelingDQN, self).__init__()

        # 共享特征层
        self.feature_layer = nn.Sequential(
            nn.Linear(state_dim, hidden_dims[0]),
            nn.ReLU(inplace=True),
        )

        # 价值流 V(s)
        self.value_stream = nn.Sequential(
            nn.Linear(hidden_dims[0], hidden_dims[1]),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dims[1], 1),  # 输出单个价值
        )

        # 优势流 A(s, a)
        self.advantage_stream = nn.Sequential(
            nn.Linear(hidden_dims[0], hidden_dims[1]),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dims[1], action_dim),  # 输出每个动作的优势
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, state):
        features = self.feature_layer(state)
        value = self.value_stream(features)
        advantage = self.advantage_stream(features)
        # Q(s,a) = V(s) + A(s,a) - mean(A(s,a))
        return value + advantage - advantage.mean(dim=1, keepdim=True)


# ============================================================================
# 第三部分：经验回放缓冲区
# ============================================================================

class ReplayBuffer:
    """
    经验回放缓冲区 (Experience Replay Buffer)

    存储智能体与环境交互产生的转移样本 (s, a, r, s', done)，
    训练时从中随机采样小批量数据，打破样本之间的时序相关性。
    """

    def __init__(self, capacity=10000):
        """
        参数:
            capacity: 缓冲区最大容量
        """
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        """存入一条经验"""
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        """随机采样一个批次"""
        batch = random.sample(self.buffer, min(batch_size, len(self.buffer)))
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            torch.FloatTensor(np.array(states)),
            torch.LongTensor(np.array(actions)).unsqueeze(1),
            torch.FloatTensor(np.array(rewards)).unsqueeze(1),
            torch.FloatTensor(np.array(next_states)),
            torch.FloatTensor(np.array(dones)).unsqueeze(1),
        )

    def __len__(self):
        return len(self.buffer)


# ============================================================================
# 第四部分：DQN 智能体
# ============================================================================

class DQNAgent:
    """
    DQN 智能体

    关键技术:
    1. 经验回放 (Experience Replay) — 随机采样打破时序相关性
    2. 目标网络 (Target Network) — 稳定训练目标
    3. ε-贪心策略 (ε-greedy) — 平衡探索与利用
    4. Double DQN — 使用主网络选择动作，目标网络评估价值，减少过高估计
    """

    def __init__(self, state_dim=2, action_dim=4,
                 learning_rate=0.001, gamma=0.99,
                 epsilon_start=1.0, epsilon_end=0.01, epsilon_decay=0.995,
                 buffer_capacity=10000, batch_size=64,
                 target_update_freq=100,
                 use_double_dqn=True, use_dueling=True,
                 device=None):
        """
        参数:
            state_dim: 状态维度
            action_dim: 动作维度
            learning_rate: 学习率
            gamma: 折扣因子
            epsilon_start: 初始探索率
            epsilon_end: 最低探索率
            epsilon_decay: 探索率衰减系数
            buffer_capacity: 经验池容量
            batch_size: 批量大小
            target_update_freq: 目标网络更新频率（步数）
            use_double_dqn: 是否使用 Double DQN
            use_dueling: 是否使用 Dueling DQN 网络
            device: 计算设备
        """
        self.action_dim = action_dim
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        self.use_double_dqn = use_double_dqn
        self.use_dueling = use_dueling

        # 设备
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device

        # 网络: 主网络 Q(s,a;θ) 和目标网络 Q(s,a;θ⁻)
        if use_dueling:
            self.q_network = DuelingDQN(state_dim, action_dim).to(self.device)
            self.target_network = DuelingDQN(state_dim, action_dim).to(self.device)
        else:
            self.q_network = DQN(state_dim, action_dim).to(self.device)
            self.target_network = DQN(state_dim, action_dim).to(self.device)

        self.target_network.load_state_dict(self.q_network.state_dict())
        self.target_network.eval()  # 目标网络不训练

        self.optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate)
        self.loss_fn = nn.MSELoss()

        # 经验回放缓冲区
        self.replay_buffer = ReplayBuffer(capacity=buffer_capacity)

        # 训练计数器
        self.train_steps = 0

        print(f"[DQN Agent] 设备: {self.device}")
        print(f"[DQN Agent] Double DQN: {use_double_dqn}, Dueling: {use_dueling}")

    def select_action(self, state, evaluate=False):
        """
        使用 ε-贪心策略选择动作

        参数:
            state: 当前状态
            evaluate: True 时使用纯贪心策略（评估模式）

        返回:
            选择的动作 (0~3)
        """
        if not evaluate and np.random.random() < self.epsilon:
            return np.random.randint(self.action_dim)

        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_values = self.q_network(state_tensor)
        return q_values.argmax(dim=1).item()

    def update(self):
        """执行一步 DQN 更新（从经验池采样并训练）"""
        if len(self.replay_buffer) < self.batch_size:
            return None  # 经验不足，暂不训练

        self.train_steps += 1

        # 从经验池采样
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(
            self.batch_size)
        states = states.to(self.device)
        actions = actions.to(self.device)
        rewards = rewards.to(self.device)
        next_states = next_states.to(self.device)
        dones = dones.to(self.device)

        # 计算当前 Q 值
        current_q = self.q_network(states).gather(1, actions)

        # 计算目标 Q 值
        with torch.no_grad():
            if self.use_double_dqn:
                # Double DQN: 主网络选择动作，目标网络评估价值
                next_actions = self.q_network(next_states).argmax(dim=1, keepdim=True)
                next_q = self.target_network(next_states).gather(1, next_actions)
            else:
                # 标准 DQN: 目标网络直接取 max
                next_q = self.target_network(next_states).max(dim=1, keepdim=True).values

            target_q = rewards + self.gamma * next_q * (1 - dones)

        # 计算损失并更新
        loss = self.loss_fn(current_q, target_q)
        self.optimizer.zero_grad()
        loss.backward()
        # 梯度裁剪，防止梯度爆炸
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), max_norm=10.0)
        self.optimizer.step()

        # 定期同步目标网络
        if self.train_steps % self.target_update_freq == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())

        # ε 衰减
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)

        return loss.item()

    def save(self, path):
        """保存模型"""
        torch.save({
            'q_network': self.q_network.state_dict(),
            'target_network': self.target_network.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'train_steps': self.train_steps,
        }, path)
        print(f"[模型] 已保存至 {path}")

    def load(self, path):
        """加载模型"""
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        self.q_network.load_state_dict(checkpoint['q_network'])
        self.target_network.load_state_dict(checkpoint['target_network'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.epsilon = checkpoint['epsilon']
        self.train_steps = checkpoint['train_steps']
        print(f"[模型] 已从 {path} 加载")


# ============================================================================
# 第五部分：训练循环与评估
# ============================================================================

def train(env, agent, num_episodes=800, log_interval=50, early_stop_window=100,
          early_stop_threshold=0.9):
    """
    训练 DQN 智能体

    参数:
        env: 仓库环境
        agent: DQN 智能体
        num_episodes: 训练回合数
        log_interval: 日志输出间隔
        early_stop_window: 早停窗口大小
        early_stop_threshold: 早停成功率阈值

    返回:
        metrics: 训练指标字典
    """
    metrics = {
        'episode_rewards': [],
        'episode_steps': [],
        'episode_success': [],
        'episode_losses': [],
        'success_rate': [],  # 滑动窗口成功率
    }

    print(f"\n{'='*60}")
    print(f"开始训练 | 总回合数: {num_episodes} | 早停阈值: {early_stop_threshold}")
    print(f"{'='*60}\n")

    for episode in range(1, num_episodes + 1):
        state = env.reset()
        episode_reward = 0
        episode_loss = 0
        loss_count = 0

        while True:
            # 选择动作
            action = agent.select_action(state)

            # 执行动作
            next_state, reward, done, info = env.step(action)

            # 存入经验池
            agent.replay_buffer.push(state, action, reward, next_state, done)

            # DQN 更新
            loss = agent.update()
            if loss is not None:
                episode_loss += loss
                loss_count += 1

            state = next_state
            episode_reward += reward

            if done:
                break

        # 记录指标
        metrics['episode_rewards'].append(episode_reward)
        metrics['episode_steps'].append(env.steps)
        metrics['episode_success'].append(int(env.success))
        metrics['episode_losses'].append(
            episode_loss / max(loss_count, 1) if loss_count > 0 else 0)

        # 计算滑动窗口成功率
        window = metrics['episode_success'][-early_stop_window:]
        success_rate = sum(window) / len(window)
        metrics['success_rate'].append(success_rate)

        # 日志输出
        if episode % log_interval == 0 or episode == 1:
            recent_50 = sum(metrics['episode_success'][-50:]) / min(50, episode)
            print(f"回合 {episode:4d}/{num_episodes} | "
                  f"奖励: {episode_reward:7.1f} | "
                  f"步数: {env.steps:3d} | "
                  f"ε: {agent.epsilon:.3f} | "
                  f"近50回合成功率: {recent_50:.2%} | "
                  f"缓冲区: {len(agent.replay_buffer)}")

        # 早停检查
        if episode >= early_stop_window and success_rate >= early_stop_threshold:
            print(f"\n[早停] 回合 {episode}: 近{early_stop_window}回合成功率 "
                  f"达到 {success_rate:.2%}，停止训练")
            break

    print(f"\n{'='*60}")
    print(f"训练完成 | 总回合数: {episode}")
    print(f"最终近{early_stop_window}回合成功率: {success_rate:.2%}")
    print(f"{'='*60}")

    return metrics


def evaluate(env, agent, num_episodes=100, render_every=20):
    """
    评估训练后的智能体（纯贪心策略，不探索）

    参数:
        env: 仓库环境
        agent: DQN 智能体
        num_episodes: 评估回合数
        render_every: 每隔多少回合渲染一次路径图

    返回:
        eval_results: 评估结果字典
    """
    successes = 0
    total_steps = 0
    all_paths = []
    failure_cases = []

    for episode in range(1, num_episodes + 1):
        state = env.reset()
        episode_reward = 0

        while True:
            action = agent.select_action(state, evaluate=True)  # 纯贪心
            next_state, reward, done, info = env.step(action)
            state = next_state
            episode_reward += reward
            if done:
                break

        if env.success:
            successes += 1
            total_steps += env.steps
            all_paths.append(list(env.path_history))
        else:
            failure_cases.append({
                'episode': episode,
                'reason': info.get('reason', 'unknown'),
                'steps': env.steps,
                'reward': episode_reward,
                'path': list(env.path_history),
            })

        # 渲染典型路径
        if episode % render_every == 0 or (episode <= 5 and env.success):
            fig, ax = plt.subplots(figsize=(6, 6))
            env.render_matplotlib(ax, show_path=True,
                                  title=f"评估回合 {episode}: {'成功 ✓' if env.success else '失败 ✗'}")
            plt.tight_layout()
            plt.show()

    accuracy = successes / num_episodes
    avg_steps = total_steps / max(successes, 1)

    print(f"\n{'='*60}")
    print(f"评估结果 ({num_episodes} 回合, 纯贪心策略)")
    print(f"  成功率: {accuracy:.2%} ({successes}/{num_episodes})")
    print(f"  平均成功步数: {avg_steps:.1f}")
    print(f"  失败案例数: {len(failure_cases)}")
    if failure_cases:
        print(f"  失败原因分布:")
        reasons = {}
        for f in failure_cases:
            reasons[f['reason']] = reasons.get(f['reason'], 0) + 1
        for reason, count in reasons.items():
            print(f"    - {reason}: {count} 次")
    print(f"{'='*60}")

    return {
        'accuracy': accuracy,
        'avg_steps': avg_steps,
        'successes': successes,
        'total': num_episodes,
        'paths': all_paths,
        'failures': failure_cases,
    }


# ============================================================================
# 第六部分：可视化工具
# ============================================================================

def plot_training_curves(metrics, save_path=None):
    """
    绘制训练曲线: 奖励曲线、成功率曲线、步数曲线、损失曲线
    """
    episodes = range(1, len(metrics['episode_rewards']) + 1)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 子图1: 每回合奖励
    ax1 = axes[0, 0]
    ax1.plot(episodes, metrics['episode_rewards'], alpha=0.3, color='blue', linewidth=0.5)
    # 滑动平均
    window = min(50, len(metrics['episode_rewards']))
    if window > 1:
        smoothed = np.convolve(metrics['episode_rewards'],
                               np.ones(window)/window, mode='valid')
        ax1.plot(range(window, len(metrics['episode_rewards']) + 1),
                 smoothed, color='blue', linewidth=2, label=f'滑动平均 (窗口={window})')
    ax1.set_xlabel('回合数')
    ax1.set_ylabel('总奖励')
    ax1.set_title('每回合总奖励')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # 子图2: 成功率
    ax2 = axes[0, 1]
    success_array = np.array(metrics['episode_success'])
    # 滑动窗口成功率
    if len(success_array) >= window:
        smoothed_sr = np.convolve(success_array, np.ones(window)/window, mode='valid')
        ax2.plot(range(window, len(success_array) + 1),
                 smoothed_sr, color='green', linewidth=2, label=f'滑动成功率 (窗口={window})')
    ax2.axhline(y=0.9, color='red', linestyle='--', alpha=0.5, label='90% 参考线')
    ax2.set_xlabel('回合数')
    ax2.set_ylabel('成功率')
    ax2.set_title('成功率 (滑动窗口)')
    ax2.set_ylim(-0.05, 1.05)
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # 子图3: 步数
    ax3 = axes[1, 0]
    ax3.plot(episodes, metrics['episode_steps'], alpha=0.3, color='purple', linewidth=0.5)
    if window > 1:
        smoothed_steps = np.convolve(metrics['episode_steps'],
                                     np.ones(window)/window, mode='valid')
        ax3.plot(range(window, len(metrics['episode_steps']) + 1),
                 smoothed_steps, color='purple', linewidth=2, label=f'滑动平均 (窗口={window})')
    ax3.set_xlabel('回合数')
    ax3.set_ylabel('步数')
    ax3.set_title('每回合步数')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # 子图4: 损失
    ax4 = axes[1, 1]
    ax4.plot(episodes, metrics['episode_losses'], alpha=0.3, color='red', linewidth=0.5)
    if window > 1:
        smoothed_loss = np.convolve(metrics['episode_losses'],
                                    np.ones(window)/window, mode='valid')
        ax4.plot(range(window, len(metrics['episode_losses']) + 1),
                 smoothed_loss, color='red', linewidth=2, label=f'滑动平均 (窗口={window})')
    ax4.set_xlabel('回合数')
    ax4.set_ylabel('损失')
    ax4.set_title('训练损失')
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    plt.suptitle('DQN 仓储机器人训练曲线', fontsize=14, fontweight='bold')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"[图表] 训练曲线已保存至 {save_path}")

    plt.show()


def plot_path_comparison(env, agent, num_examples=4, save_path=None):
    """
    绘制多条成功路径的对比图
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 12))
    axes = axes.flatten()

    for i in range(num_examples):
        state = env.reset()
        while True:
            action = agent.select_action(state, evaluate=True)
            next_state, reward, done, info = env.step(action)
            state = next_state
            if done:
                break

        ax = axes[i]
        env.render_matplotlib(ax, show_path=True,
                              title=f"成功路径 {i+1}: {env.steps} 步")

    plt.suptitle('DQN 智能体典型成功路径', fontsize=14, fontweight='bold')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')

    plt.show()


def plot_failure_analysis(env, agent, save_path=None):
    """分析并可视化失败案例"""
    # 收集失败案例
    failure_paths = []
    attempts = 0
    max_attempts = 500

    while len(failure_paths) < 4 and attempts < max_attempts:
        state = env.reset()
        while True:
            action = agent.select_action(state, evaluate=True)
            next_state, reward, done, info = env.step(action)
            state = next_state
            if done:
                break
        attempts += 1

        if not env.success:
            failure_paths.append({
                'path': list(env.path_history),
                'reason': info.get('reason', 'unknown'),
                'steps': env.steps,
            })

    if not failure_paths:
        print("[分析] 未发现失败案例，智能体表现良好！")
        return

    fig, axes = plt.subplots(2, 2, figsize=(12, 12))
    axes = axes.flatten()

    for i, case in enumerate(failure_paths[:4]):
        # 重放路径
        env.reset()
        # 手动设置路径用于可视化
        env.path_history = case['path']
        env.robot_pos = list(case['path'][-1])

        ax = axes[i]
        reason_cn = {
            'out_of_bounds': '走出边界',
            'collision': '撞到障碍物',
            'timeout': '超过最大步数',
        }.get(case['reason'], case['reason'])

        env.render_matplotlib(ax, show_path=True,
                              title=f"失败案例 {i+1}: {reason_cn}\n步数: {case['steps']}")

        # 标记失败位置
        fail_pos = case['path'][-1]
        ax.plot(fail_pos[0], fail_pos[1], 'x', color='red', markersize=20,
                markeredgewidth=3, zorder=10)

    plt.suptitle('DQN 智能体失败案例分析', fontsize=14, fontweight='bold')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')

    plt.show()


def plot_q_value_heatmap(env, agent, save_path=None):
    """
    绘制 Q 值热力图 — 展示智能体在各位置对各动作的偏好
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 12))
    action_names = ['上 (Up)', '下 (Down)', '左 (Left)', '右 (Right)']

    for a in range(4):
        ax = axes[a // 2, a % 2]
        q_grid = np.zeros((env.grid_size, env.grid_size))

        for x in range(env.grid_size):
            for y in range(env.grid_size):
                if env.grid[x, y] == 1:  # 障碍物
                    q_grid[x, y] = np.nan
                else:
                    state = np.array([x / max(env.grid_size-1, 1),
                                     y / max(env.grid_size-1, 1)], dtype=np.float32)
                    state_tensor = torch.FloatTensor(state).unsqueeze(0).to(agent.device)
                    with torch.no_grad():
                        q_values = agent.q_network(state_tensor).cpu().numpy()[0]
                    q_grid[x, y] = q_values[a]

        im = ax.imshow(q_grid.T, origin='lower', cmap='RdYlGn',
                       extent=[-0.5, env.grid_size-0.5, -0.5, env.grid_size-0.5])
        ax.set_title(f'Q(s, {action_names[a]})')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        plt.colorbar(im, ax=ax, shrink=0.8)

        # 标记起点和目标
        ax.plot(env.start_pos[0], env.start_pos[1], 'bs', markersize=12, label='起点')
        for g in env.goals:
            ax.plot(g[0], g[1], 'g*', markersize=15, label='目标')

    plt.suptitle('Q 值热力图: 各位置的动作价值分布', fontsize=14, fontweight='bold')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')

    plt.show()


# ============================================================================
# 第七部分：超参数对比实验（加分项）
# ============================================================================

def run_hyperparameter_experiment():
    """
    对比不同超参数/改进方法的训练效果

    对比组:
        1. 基础 DQN (无改进)
        2. Double DQN
        3. Dueling DQN
        4. Double + Dueling DQN
    """
    print("\n" + "="*70)
    print("超参数对比实验")
    print("="*70)

    # 创建固定环境以保证公平对比
    np.random.seed(42)
    torch.manual_seed(42)
    random.seed(42)

    env_config = {
        'grid_size': 10,
        'obstacle_ratio': 0.12,
        'num_goals': 1,
    }

    variants = [
        {'name': '基础 DQN', 'double': False, 'dueling': False},
        {'name': 'Double DQN', 'double': True, 'dueling': False},
        {'name': 'Dueling DQN', 'double': False, 'dueling': True},
        {'name': 'Double + Dueling DQN', 'double': True, 'dueling': True},
    ]

    results = {}
    colors = ['blue', 'orange', 'green', 'red']

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    for variant, color in zip(variants, colors):
        print(f"\n--- 训练: {variant['name']} ---")

        np.random.seed(42)
        torch.manual_seed(42)
        random.seed(42)

        env = WarehouseGridEnv(**env_config)
        agent = DQNAgent(
            use_double_dqn=variant['double'],
            use_dueling=variant['dueling'],
            epsilon_decay=0.995,
        )

        metrics = train(env, agent, num_episodes=500, log_interval=100,
                        early_stop_window=100, early_stop_threshold=0.95)

        results[variant['name']] = metrics

        window = 30
        episodes = range(1, len(metrics['episode_rewards']) + 1)

        # 绘制对比曲线
        ax1 = axes[0, 0]
        if len(metrics['episode_rewards']) > window:
            smoothed = np.convolve(metrics['episode_rewards'],
                                   np.ones(window)/window, mode='valid')
            ax1.plot(range(window, len(metrics['episode_rewards']) + 1),
                     smoothed, color=color, linewidth=2, label=variant['name'])

        ax2 = axes[0, 1]
        if len(metrics['episode_success']) > window:
            smoothed_sr = np.convolve(metrics['episode_success'],
                                      np.ones(window)/window, mode='valid')
            ax2.plot(range(window, len(metrics['episode_success']) + 1),
                     smoothed_sr, color=color, linewidth=2, label=variant['name'])

        ax3 = axes[1, 0]
        if len(metrics['episode_steps']) > window:
            smoothed_st = np.convolve(metrics['episode_steps'],
                                      np.ones(window)/window, mode='valid')
            ax3.plot(range(window, len(metrics['episode_steps']) + 1),
                     smoothed_st, color=color, linewidth=2, label=variant['name'])

        ax4 = axes[1, 1]
        if len(metrics['episode_losses']) > window:
            smoothed_l = np.convolve(metrics['episode_losses'],
                                     np.ones(window)/window, mode='valid')
            ax4.plot(range(window, len(metrics['episode_losses']) + 1),
                     smoothed_l, color=color, linewidth=2, label=variant['name'])

    axes[0, 0].set_title('奖励对比 (滑动平均)')
    axes[0, 0].set_xlabel('回合数'); axes[0, 0].set_ylabel('总奖励')
    axes[0, 0].legend(); axes[0, 0].grid(True, alpha=0.3)

    axes[0, 1].set_title('成功率对比 (滑动平均)')
    axes[0, 1].set_xlabel('回合数'); axes[0, 1].set_ylabel('成功率')
    axes[0, 1].legend(); axes[0, 1].grid(True, alpha=0.3)

    axes[1, 0].set_title('步数对比 (滑动平均)')
    axes[1, 0].set_xlabel('回合数'); axes[1, 0].set_ylabel('步数')
    axes[1, 0].legend(); axes[1, 0].grid(True, alpha=0.3)

    axes[1, 1].set_title('损失对比 (滑动平均)')
    axes[1, 1].set_xlabel('回合数'); axes[1, 1].set_ylabel('损失')
    axes[1, 1].legend(); axes[1, 1].grid(True, alpha=0.3)

    plt.suptitle('DQN 变体对比实验', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('hyperparameter_comparison.png', dpi=150, bbox_inches='tight')
    plt.show()

    return results


# ============================================================================
# 第八部分：主程序入口
# ============================================================================

def main():
    """
    一键运行入口

    执行流程:
        1. 创建仓库环境并可视化
        2. 创建 DQN 智能体
        3. 训练智能体
        4. 绘制训练曲线
        5. 评估智能体并可视化路径
        6. 分析失败案例
        7. 绘制 Q 值热力图
    """
    print("="*60)
    print("基于 DQN 的智能仓储机器人路径规划与货架拣选策略")
    print("="*60)

    # ---- 配置参数 ----
    GRID_SIZE = 10
    OBSTACLE_RATIO = 0.12
    NUM_GOALS = 1
    NUM_EPISODES = 800
    LEARNING_RATE = 0.001
    GAMMA = 0.99
    EPSILON_START = 1.0
    EPSILON_END = 0.01
    EPSILON_DECAY = 0.995
    BUFFER_CAPACITY = 10000
    BATCH_SIZE = 64
    TARGET_UPDATE_FREQ = 100
    USE_DOUBLE_DQN = True   # 使用 Double DQN 改进
    USE_DUELING = True      # 使用 Dueling DQN 网络

    # ---- 设置随机种子（保证可复现性） ----
    SEED = 42
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    random.seed(SEED)

    # ---- 第1步: 创建环境 ----
    print("\n[1/7] 创建仓库网格环境...")
    env = WarehouseGridEnv(
        grid_size=GRID_SIZE,
        obstacle_ratio=OBSTACLE_RATIO,
        num_goals=NUM_GOALS,
    )

    print(f"  网格大小: {GRID_SIZE}x{GRID_SIZE}")
    print(f"  障碍物比例: {OBSTACLE_RATIO}")
    print(f"  起点: {env.start_pos}")
    print(f"  目标货架: {env.goals}")
    print(f"  障碍物数量: {np.sum(env.grid == 1)}")

    # 可视化初始环境
    fig, ax = plt.subplots(figsize=(6, 6))
    env.render_matplotlib(ax, title="仓库网格环境 (初始状态)")
    plt.tight_layout()
    plt.savefig('01_environment.png', dpi=150, bbox_inches='tight')
    plt.show()

    # 文本渲染
    print("\n  环境文本渲染:")
    env.render_text()

    # ---- 第2步: 创建 DQN 智能体 ----
    print("\n[2/7] 创建 DQN 智能体...")
    agent = DQNAgent(
        state_dim=2,
        action_dim=4,
        learning_rate=LEARNING_RATE,
        gamma=GAMMA,
        epsilon_start=EPSILON_START,
        epsilon_end=EPSILON_END,
        epsilon_decay=EPSILON_DECAY,
        buffer_capacity=BUFFER_CAPACITY,
        batch_size=BATCH_SIZE,
        target_update_freq=TARGET_UPDATE_FREQ,
        use_double_dqn=USE_DOUBLE_DQN,
        use_dueling=USE_DUELING,
    )

    # ---- 第3步: 训练 ----
    print("\n[3/7] 开始训练 DQN 智能体...")
    metrics = train(env, agent, num_episodes=NUM_EPISODES,
                    log_interval=50, early_stop_window=100,
                    early_stop_threshold=0.9)

    # 保存模型
    agent.save('warehouse_dqn_model.pt')

    # ---- 第4步: 绘制训练曲线 ----
    print("\n[4/7] 绘制训练曲线...")
    plot_training_curves(metrics, save_path='02_training_curves.png')

    # ---- 第5步: 评估与路径可视化 ----
    print("\n[5/7] 评估智能体并可视化路径...")
    eval_results = evaluate(env, agent, num_episodes=100, render_every=25)

    # 多条成功路径对比
    plot_path_comparison(env, agent, num_examples=4,
                         save_path='03_successful_paths.png')

    # ---- 第6步: 失败案例分析 ----
    print("\n[6/7] 分析失败案例...")
    plot_failure_analysis(env, agent, save_path='04_failure_analysis.png')

    # ---- 第7步: Q 值热力图 ----
    print("\n[7/7] 绘制 Q 值热力图...")
    plot_q_value_heatmap(env, agent, save_path='05_q_value_heatmap.png')

    # ---- 输出总结 ----
    print("\n" + "="*60)
    print("实验总结")
    print("="*60)
    print(f"  环境: {GRID_SIZE}x{GRID_SIZE} 网格, "
          f"障碍物比例 {OBSTACLE_RATIO}, {NUM_GOALS} 个目标")
    print(f"  算法: {'Double ' if USE_DOUBLE_DQN else ''}"
          f"{'Dueling ' if USE_DUELING else ''}DQN")
    print(f"  超参数: lr={LEARNING_RATE}, gamma={GAMMA}, "
          f"epsilon_decay={EPSILON_DECAY}")
    print(f"  训练回合数: {len(metrics['episode_rewards'])}")
    print(f"  最终滑动成功率: {metrics['success_rate'][-1]:.2%}")
    print(f"  评估成功率: {eval_results['accuracy']:.2%}")
    print(f"  评估平均步数: {eval_results['avg_steps']:.1f}")
    print(f"\n  生成文件:")
    print(f"    - 01_environment.png  (初始环境)")
    print(f"    - 02_training_curves.png  (训练曲线)")
    print(f"    - 03_successful_paths.png  (成功路径)")
    print(f"    - 04_failure_analysis.png  (失败分析)")
    print(f"    - 05_q_value_heatmap.png  (Q值热力图)")
    print(f"    - warehouse_dqn_model.pt  (模型权重)")
    print("="*60)

    return env, agent, metrics, eval_results


# ============================================================================
# 程序入口
# ============================================================================

if __name__ == '__main__':
    # 检查依赖
    missing_deps = []
    try:
        import numpy
    except ImportError:
        missing_deps.append('numpy')
    try:
        import matplotlib
    except ImportError:
        missing_deps.append('matplotlib')
    try:
        import torch
    except ImportError:
        missing_deps.append('torch')

    if missing_deps:
        print(f"缺少依赖: {', '.join(missing_deps)}")
        print(f"请运行: pip install {' '.join(missing_deps)}")
        exit(1)

    # ---- 选择运行模式 ----
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == '--compare':
        # 超参数对比模式
        run_hyperparameter_experiment()
    else:
        # 标准训练模式
        env, agent, metrics, eval_results = main()
