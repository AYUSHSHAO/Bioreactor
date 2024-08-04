import numpy as np
import gym
from gym import spaces
tanh = np.tanh
import matplotlib.pyplot as plt
import math
import torch
pow = math.pow
exp = np.exp
import torch.nn as nn
import torch.optim as optim
from torch import Tensor
from torch.nn import functional
import copy
from math import exp
import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
global dt
dt = 60



def get_state(action, ti, x0):
    flowrate = action  # ml/day
    gf = 18  # mg/ml total inlet clucose(20)
    T = 308  # K temperature(310)
    protein_ref = 590  # 590
    viability_ref = 96.45

    def model(y, t):
        v = y[0]  # ml  volume of mixture
        x = y[1]  # 10^6 viable cells/ml
        p = y[2]  # mg/l mab conc.
        s = y[3]  # g/l glucose conc.
        l = y[4]  # g/l lactate conc
        vb = y[5]  # no unit viability

        qif = flowrate  # ml/min
        qos = flowrate  # ml/min

        parameters = [1.53065905e-01, 8.50356543e-01, 3.95525447e-05, 6.61966255e-05,
                      1.20866880e-02, 2.05140571e+00, 1.10167094e-04, 3.44210432e+02,
                      1.98322526e+00, 9.33512253e-02, 6.87871174e-04, 6.28306289e+03,
                      4.99990345e-01, 1.88752322e-05, 1.68890312e+03, 5.24079503e+02]
        mumax, ks, mud, ypx, mp, yxs, ms, kl, yxl, yls, kp, lacmax1, lacmax2, mlac, k1, k2 = parameters
        xd = x * (100 / vb - 1)
        if (ks * x + s) * (l + kl) != 0:
            mu = mumax * s * kl * (1 - kp * p) * exp(-k1 / T) / ((ks * x + s) * (l + kl))
        else:
            mu = mumax * s * kl * (1 - kp * p) * exp(-k1 / T) / ((ks * x + s + 0.0000001) * (l + kl + 0.000001))
        dvdt = qif - qos
        dxvdt = (mu - mud * exp(-k2 / T)) * x * v - qos * x  # Xv
        dpvdt = (ypx * mu + mp) * x * v - qos * p  # mAb
        dsvdt = qif * gf - (qos) * s - (mu / yxs + ms) * x * v  # glucose
        dlvdt = (mu / yxl + yls * (mu / yxs + ms) * (lacmax1 - l) / lacmax1 + mlac * (
                    lacmax2 - l) / lacmax2) * x * v - (qos) * l  # lac
        dxdvdt = mud * exp(-k2 / T) * x * v - xd * qos  # Xd
        dxdt = (dxvdt - x * dvdt) / v  # divided by V (conc)
        dpdt = (dpvdt - p * dvdt) / v
        dsdt = (dsvdt - s * dvdt) / v
        dldt = (dlvdt - l * dvdt) / v
        dxddt = (dxdvdt - xd * dvdt) / v
        dvbdt = 100 * ((x + xd) * dxdt - x * (dxdt + dxddt)) / ((x + xd) * (x + xd))
        return [dvdt[0], dxdt[0], dpdt[0], dsdt[0], dldt[0], dvbdt[0]]

    tspan = np.linspace(ti, ti + dt, 10)
    y = odeint(model, x0, tspan)
    All = y[-1]
    P = All[2]
    A = y[9, 0]
    B = y[9, 1]
    C = y[9, 2]
    D = y[9, 3]
    E = y[9, 4]
    F = y[9, 5]
    #     rewards=-20*(np.abs(All[2]-protein_ref))-10*(np.abs(All[5]-viability_ref))-10*action
    rewards = -(abs(All[2] - 590))
    return All, rewards

x0 = [5400, 4.147507600512498, 107.96076361017765, 2.614975072822183, 1.8767447491163762, 98.31311438674405]

# seed=50

seed = 100
torch.manual_seed(seed)



def plot_G(protein_ep, tot_time, flowrate_ep, name):
    time = np.linspace(0, tot_time, int(tot_time / dt))
    T1 = 590  # target
    ta = np.ones(int(tot_time / dt)) * T1
    fig, ax1 = plt.subplots()
    # time = np.linspace(0,T,int(T/dt))
    font1 = {'family': 'serif', 'size': 15}
    font2 = {'family': 'serif', 'size': 15}
    color = 'tab:red'
    ax1.set_xlabel('time (min)', fontdict=font1)
    ax1.set_ylabel('Protein Concentration', fontdict=font2, color=color)
    ax1.plot(time, protein_ep, color=color)
    ax1.plot(time, ta, color='tab:orange', linewidth=4, label='mAB reference concentration')
    leg = ax1.legend(loc='lower right')

    ax1.tick_params(axis='y', labelcolor=color)

    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
    color = 'tab:blue'
    ax2.set_ylabel('flowrate', fontdict=font2, color=color)  # we already handled the x-label with ax1
    ax2.step(time, flowrate_ep, where='post', color=color)
    ax2.tick_params(axis='y', labelcolor=color)
    ax2.grid(color='g', linestyle='-', linewidth=1)

    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    # plt.savefig('deeprl_test1_batch1.jpg')
    plt.savefig(name + '.png')
    plt.close()

class ReplayBuffer_Low:
    def __init__(self, max_size=50000):  # max_size=5e5
        self.buffer = []
        self.max_size = int(max_size)
        self.size = 0

    def add(self, transition):
        assert len(transition) == 7, "transition must have length = 6"

        # transiton is tuple of (state, action, reward, next_state, goal, gamma)
        self.buffer.append(transition)
        self.size += 1

    def sample(self, batch_size):
        # delete 1/5th of the buffer when full
        if self.size > self.max_size:
            del self.buffer[0:int(self.size / 5)]
            self.size = len(self.buffer)
        # print(len(self.buffer))
        indexes = np.random.randint(0, len(self.buffer), size=batch_size)
        states, actions, goals, rewards, next_states, next_goals, gamma = [], [], [], [], [], [], []

        for i in indexes:
            states.append(np.array(self.buffer[i][0], copy=False))
            actions.append(np.array(self.buffer[i][1], copy=False))
            goals.append(np.array(self.buffer[i][2], copy=False))
            rewards.append(np.array(self.buffer[i][3], copy=False))
            next_states.append(np.array(self.buffer[i][4], copy=False))
            next_goals.append(np.array(self.buffer[i][5], copy=False))
            gamma.append(np.array(self.buffer[i][6], copy=False))

        return np.array(states), np.array(actions), np.array(goals), np.array(rewards), np.array(next_states), np.array(
            next_goals), np.array(gamma)


class ReplayBuffer_High:
    def __init__(self, max_size=50000):  # max_size=5e5
        self.buffer = []
        self.max_size = int(max_size)
        self.size = 0

    def add(self, transition):
        assert len(transition) == 5, "transition must have length = 5"

        # transiton is tuple of (state, action, reward, next_state, goal, gamma)
        self.buffer.append(transition)
        self.size += 1

    def sample(self, batch_size):
        # delete 1/5th of the buffer when full
        if self.size > self.max_size:
            del self.buffer[0:int(self.size / 5)]
            self.size = len(self.buffer)
        # print(len(self.buffer))
        indexes = np.random.randint(0, len(self.buffer), size=batch_size)
        states, goals, rewards, next_states, gamma = [], [], [], [], []

        for i in indexes:
            states.append(np.array(self.buffer[i][0], copy=False))
            goals.append(np.array(self.buffer[i][1], copy=False))
            rewards.append(np.array(self.buffer[i][2], copy=False))
            next_states.append(np.array(self.buffer[i][3], copy=False))
            gamma.append(np.array(self.buffer[i][4], copy=False))

        return np.array(states), np.array(goals), np.array(rewards), np.array(next_states), np.array(gamma)


class Actor_Low(nn.Module):
    def __init__(self, state_dim, action_dim, goal_dim, action_bound, action_offset):
        super(Actor_Low, self).__init__()
        # actor
        self.actor = nn.Sequential(
            nn.Linear(state_dim + goal_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim),
            nn.ReLU()
        )
        # max value of actions

        self.offset = action_offset
        self.bounds = action_bound

    def forward(self, state, goal):

        #return self.actor(torch.cat([state, goal], 1))

        return self.actor(torch.cat([state, goal], 1))#*self.bounds + self.offset


class Actor_High(nn.Module):
    def __init__(self, state_dim, goal_dim, goal_index, state_bound, state_offset): # goal_dim as goal is the action taken by the higher level policy
        super(Actor_High, self).__init__()
        # actor
        self.actor = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, goal_dim),
            nn.ReLU()
        )
        # max value of actions
        self.offset = state_offset
        self.bounds = state_bound
        self.offset = state_offset[:,goal_index]
        self.bounds = state_bound[:,goal_index]

    def forward(self, state):
        return self.actor(state)#*self.bounds + self.offset


class Critic_Low(nn.Module):
    def __init__(self, state_dim, action_dim, goal_dim):
        super(Critic_Low, self).__init__()
        # UVFA critic
        self.critic = nn.Sequential(
            nn.Linear(state_dim + action_dim + goal_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, state, action, goal):
        return self.critic(torch.cat([state, action, goal], 1))


class Critic_High(nn.Module):
    def __init__(self, state_dim, goal_dim): # goal_dim as goal is the action taken by the higher level policy
        super(Critic_High, self).__init__()
        # UVFA critic
        self.critic = nn.Sequential(
            nn.Linear(state_dim + goal_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, state, goal):
        return self.critic(torch.cat([state, goal], 1))


class DDPG_Low:
    def __init__(self, state_dim, action_dim, goal_dim, action_bounds, action_offset, policy_freq, tau, action_policy_noise, action_policy_clip, lr):

        ################### For Lower Level Policy############################
        self.action_policy_noise = action_policy_noise
        self.action_policy_clip = action_policy_clip
        self.action_offset = action_offset
        self.action_bounds = action_bounds
        self.tau = tau
        self.policy_freq = policy_freq
        self.goal_dim = goal_dim # as goal is the state which our agent should acheive
        self.actor_Low = Actor_Low(state_dim, action_dim, self.goal_dim, self.action_bounds, self.action_offset).to(device)
        self.actor_Low_target = Actor_Low(state_dim, action_dim, self.goal_dim, self.action_bounds, self.action_offset).to(device)
        self.actor_optimizer = optim.Adam(self.actor_Low.parameters(), lr=lr)
        # two critics for TD3 learning
        self.critic_Low_1 = Critic_Low(state_dim, action_dim, self.goal_dim).to(device)
        self.critic_Low_target_1 = Critic_Low(state_dim, action_dim, self.goal_dim).to(device)
        self.critic_Low_1_optimizer = optim.Adam(self.critic_Low_1.parameters(), lr=lr)

        self.critic_Low_2 = Critic_Low(state_dim, action_dim, self.goal_dim).to(device)
        self.critic_Low_target_2 = Critic_Low(state_dim, action_dim, self.goal_dim).to(device)
        self.critic_Low_2_optimizer = optim.Adam(self.critic_Low_2.parameters(), lr=lr)

        self.mseLoss = torch.nn.MSELoss()



    def select_action_Low(self, state, goal):
        # print(type(state))
        state = torch.FloatTensor(state.reshape(1, -1)).to(device)
        goal = torch.FloatTensor(goal.reshape(1, -1)).to(device)

        return self.actor_Low(state, goal).detach().cpu().data.numpy().flatten()




    def update_Low(self, buffer, n_iter, batch_size):
        for i in range(n_iter):
            # Sample a batch of transitions from replay buffer:
            state, action, goal, reward, next_state, next_goal, gamma = buffer.sample(batch_size)

            # convert np arrays into tensors
            state = torch.FloatTensor(state).to(device)
            action = torch.FloatTensor(action).to(device)
            next_goal = torch.FloatTensor(next_goal).to(device)
            reward = torch.FloatTensor(reward).reshape((batch_size, 1)).to(device)
            next_state = torch.FloatTensor(next_state).to(device)
            goal = torch.FloatTensor(goal).to(device)
            gamma = torch.FloatTensor(gamma).reshape((batch_size, 1)).to(device)

            # select next action
            noise = action.data.normal_(0, self.action_policy_noise.item()).to(device)
            noise = noise.clamp(-torch.from_numpy(self.action_policy_clip), torch.from_numpy(self.action_policy_clip))

            next_action = self.actor_Low_target(next_state, next_goal).detach()

            next_action = (next_action + noise).clamp(self.action_offset, self.action_offset+self.action_bounds).to(torch.float32)
            # take the minimum of 2 q-values ---> TD3

            target_Q1 = self.critic_Low_target_1(next_state, next_action, next_goal).detach()
            target_Q2 = self.critic_Low_target_2(next_state, next_action, next_goal).detach()
            target_Q = torch.min(target_Q1, target_Q2)
            # Compute target Q-value:
            target_Q = reward + gamma * target_Q

            current_Q1 = self.critic_Low_1(state, action, goal)
            current_Q2 = self.critic_Low_2(state, action, goal)

            # Optimize Critic:
            critic_loss1 = self.mseLoss(current_Q1, target_Q)
            self.critic_Low_1_optimizer.zero_grad()
            critic_loss1.backward()
            self.critic_Low_1_optimizer.step()

            critic_loss2 = self.mseLoss(current_Q2, target_Q)
            self.critic_Low_2_optimizer.zero_grad()
            critic_loss2.backward()
            self.critic_Low_2_optimizer.step()



            if i % self.policy_freq == 0:

                # Compute actor loss:
                actor_loss = -self.critic_Low_1(state, self.actor_Low(state, goal), goal).mean()

                # Optimize the actor
                self.actor_optimizer.zero_grad()
                actor_loss.backward()
                self.actor_optimizer.step()

                for param, target_param in zip(self.actor_Low.parameters(), self.actor_Low_target.parameters()):
                    target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

                for param, target_param in zip(self.critic_Low_1.parameters(), self.critic_Low_target_1.parameters()):
                    target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

                for param, target_param in zip(self.critic_Low_2.parameters(), self.critic_Low_target_2.parameters()):
                    target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)




    def save(self, directory, name):
        torch.save(self.actor_Low.state_dict(), '%s/%s_actor_Low.pth' % (directory, name))
        torch.save(self.critic_Low_1.state_dict(), '%s/%s_critic_Low_1.pth' % (directory, name))
        torch.save(self.critic_Low_2.state_dict(), '%s/%s_critic_Low_2.pth' % (directory, name))


    def load(self, directory, name):
        self.actor_Low.load_state_dict(torch.load('%s/%s_actor_Low.pth' % (directory, name), map_location='cpu'))
        self.critic_Low_1.load_state_dict(torch.load('%s/%s_critic_Low_1.pth' % (directory, name), map_location='cpu'))
        self.critic_Low_2.load_state_dict(torch.load('%s/%s_critic_Low_2.pth' % (directory, name), map_location='cpu'))



class DDPG_High:
    def __init__(self, state_dim, goal_dim, goal_index, state_bound, state_offset, policy_freq, tau, state_policy_noise, state_policy_clip, lr):

        ################### For Higher Level_Policy############################
        self.state_bound = state_bound
        self.state_offset = state_offset
        self.state_policy_noise = state_policy_noise
        self.state_policy_clip = state_policy_clip
        self.policy_freq = policy_freq
        self.tau = tau
        self.goal_dim = goal_dim  # as goal is the action that higher policy will give
        self.actor_High = Actor_High(state_dim, self.goal_dim, goal_index, state_bound, state_offset).to(device)
        self.actor_High_target = Actor_High(state_dim, self.goal_dim, goal_index, state_bound, state_offset).to(device)
        self.actor_optimizer = optim.Adam(self.actor_High.parameters(), lr=lr)
        # two critics for TD3 learning
        self.critic_High_1 = Critic_High(state_dim, self.goal_dim).to(device)
        self.critic_High_target_1 = Critic_High(state_dim, self.goal_dim).to(device)
        self.critic_High_1_optimizer = optim.Adam(self.critic_High_1.parameters(), lr=lr)

        self.critic_High_target_2 = Critic_High(state_dim, self.goal_dim).to(device)
        self.critic_High_2 = Critic_High(state_dim, self.goal_dim).to(device)
        self.critic_High_2_optimizer = optim.Adam(self.critic_High_2.parameters(), lr=lr)

        self.mseLoss = torch.nn.MSELoss()



    def select_action_High(self, state):
        # print(type(state))
        state = torch.FloatTensor(state.reshape(1, -1)).to(device)
        return self.actor_High(state).detach().cpu().data.numpy().flatten()


    def update_High(self, buffer, n_iter, batch_size):
        for i in range(n_iter):
            # Sample a batch of transitions from replay buffer:
            state,  goal, reward, next_state, gamma = buffer.sample(batch_size)

            # convert np arrays into tensors
            state = torch.FloatTensor(state).to(device)
            goal = torch.FloatTensor(goal).to(device)
            reward = torch.FloatTensor(reward).reshape((batch_size, 1)).to(device)
            next_state = torch.FloatTensor(next_state).to(device)
            gamma = torch.FloatTensor(gamma).reshape((batch_size, 1)).to(device)

            # select next action

            noise = goal.data.normal_(0, self.state_policy_noise.item()).to(device)
            noise = noise.clamp(-torch.from_numpy(self.state_policy_clip), torch.from_numpy(self.state_policy_clip))

            next_goal = self.actor_High_target(next_state).detach()

            next_goal = (next_goal + noise).clamp(self.state_offset[:,self.goal_dim], self.state_offset[:,self.goal_dim] + self.state_bound[:,self.goal_dim]).to(
                torch.float32)


            # take the minimum of 2 q-values ---> TD3
            target_Q1 = self.critic_High_target_1(next_state, next_goal).detach()
            target_Q2 = self.critic_High_target_2(next_state, next_goal).detach()
            target_Q = torch.min(target_Q1, target_Q2)
            # Compute target Q-value:
            target_Q = reward + gamma * target_Q

            current_Q1 = self.critic_High_1(state, goal)
            current_Q2 = self.critic_High_2(state, goal)

            # Optimize Critic:
            critic_loss1 = self.mseLoss(current_Q1, target_Q)
            self.critic_High_1_optimizer.zero_grad()
            critic_loss1.backward()
            self.critic_High_1_optimizer.step()

            critic_loss2 = self.mseLoss(current_Q2, target_Q)
            self.critic_High_2_optimizer.zero_grad()
            critic_loss2.backward()
            self.critic_High_2_optimizer.step()


            if i % self.policy_freq == 0:

                # Compute actor loss:
                actor_loss = -self.critic_High_1(state, self.actor_High(state)).mean()

                # Optimize the actor
                self.actor_optimizer.zero_grad()
                actor_loss.backward()
                self.actor_optimizer.step()

                for param, target_param in zip(self.actor_High.parameters(), self.actor_High_target.parameters()):
                    target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

                for param, target_param in zip(self.critic_High_1.parameters(), self.critic_High_target_1.parameters()):
                    target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

                for param, target_param in zip(self.critic_High_2.parameters(), self.critic_High_target_2.parameters()):
                    target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

    def save(self, directory, name):

        torch.save(self.actor_High.state_dict(), '%s/%s_actor_High.pth' % (directory, name))
        torch.save(self.critic_High_1.state_dict(), '%s/%s_critic_High_1.pth' % (directory, name))
        torch.save(self.critic_High_2.state_dict(), '%s/%s_critic_High_2.pth' % (directory, name))

    def load(self, directory, name):

        self.actor_High.load_state_dict(torch.load('%s/%s_actor_High.pth' % (directory, name), map_location='cpu'))
        self.critic_High_1.load_state_dict(torch.load('%s/%s_critic_High_1.pth' % (directory, name), map_location='cpu'))
        self.critic_High_2.load_state_dict(torch.load('%s/%s_critic_High_2.pth' % (directory, name), map_location='cpu'))




class HAC:
    def __init__(self, k_level, policy_freq, tau, c, state_dim, action_dim, goal_dim, goal_index, goal, render,
                 threshold, action_offset, state_offset, action_bounds, state_bounds, max_goal,
                 action_policy_noise, state_policy_noise, action_policy_clip, state_policy_clip, lr):

        # adding the lowest level

        self.HAC = [DDPG_Low(state_dim, action_dim, goal_dim, action_bounds, action_offset, policy_freq, tau, action_policy_noise, action_policy_clip, lr)]
        self.replay_buffer = [ReplayBuffer_Low()]

        # adding remaining levels
        for _ in range(k_level - 1):
            self.HAC.append(DDPG_High(state_dim, goal_dim, goal_index, state_bounds, state_offset, policy_freq, tau, state_policy_noise, state_policy_clip, lr))
            self.replay_buffer.append(ReplayBuffer_High())

        # set some parameters
        self.goal_dim = goal_dim
        self.goal_index = goal_index
        self.max_goal = max_goal
        self.goal = goal
        self.k_level = k_level
        self.c = c
        self.action_dim = action_dim
        self.state_dim = state_dim
        self.threshold = threshold
        self.render = render
        self.action_bounds = action_bounds
        self.action_offset = action_offset

        # logging parameters
        self.goals = [None] * self.k_level
        self.reward = 0
        self.lo = 0
        self.iae = 0
        self.timestep = 0
        self.Protein = []
        self.viability = []
        self.flowrate = []
        self.solved = False

    def set_parameters(self, lamda, gamma, n_iter, batch_size, action_clip_low, action_clip_high,
                       state_clip_low, state_clip_high, exploration_action_noise, exploration_state_noise):

        self.lamda = lamda
        self.gamma = gamma
        self.n_iter = n_iter
        self.batch_size = batch_size
        self.action_clip_low = action_clip_low
        self.action_clip_high = action_clip_high
        self.state_clip_low = state_clip_low
        self.state_clip_high = state_clip_high
        self.exploration_action_noise = exploration_action_noise
        self.exploration_state_noise = exploration_state_noise


    def off_policy_correction(self, actor, action_sequence, state_sequence, goal_dim, goal_index, goal, end_state,
                              max_goal, device):
        # initialize
        action_sequence = torch.stack(action_sequence).to(device)
        state_sequence = torch.stack(state_sequence).to(device)
        max_goal = max_goal.cpu()
        # prepare candidates
        mean = (torch.from_numpy(end_state) - state_sequence[0])[goal_index].cpu().unsqueeze(0)
        std = 0.5 * max_goal
        candidates = [torch.min(
            torch.max(Tensor(np.random.normal(loc=mean, scale=std, size=goal_dim).astype(np.float32)), max_goal),
            -max_goal) for _ in range(3)]
        candidates.append(mean)
        candidates.append(torch.from_numpy(goal).cpu())
        # select maximal
        candidates = torch.stack(candidates).to(device)

        sequence = [state_sequence[0][goal_index] + candidate - state_sequence[:, goal_index] for candidate in
                    candidates]

        sequence = torch.stack(sequence).float().t().unsqueeze(2)
        b = state_sequence

        surr_prob = [
            -functional.mse_loss(action_sequence, actor(state_sequence.to(torch.float32), sequence[:, candidate])) for
            candidate in range(len(candidates))]

        surr_prob = [t.detach().numpy() for t in surr_prob]

        index = int(np.argmax(surr_prob))

        updated = (index != 9)
        goal_hat = candidates[index].numpy()

        return goal_hat, updated

    def intrinsic_reward(self, state, goal, next_state):
        difference = goal + state[self.goal_index] - next_state[self.goal_index]
        # difference = goal  - next_state

        distance = ((difference ** 2).sum()) ** (1 / 2)
        reward = -(distance ** 2)

        return reward  # one dimensional, difference will be the distance between states
        # return -torch.pow(sum(torch.pow(state + goal - next_state, 2)), 1 / 2)

    def dense_reward(self, state, goal):
        difference = abs(goal - state[self.goal_index])

        distance = ((difference ** 2).sum()) ** (1 / 2)
        reward = -(distance ** 2)
        # reward = -(difference.sum()) # one dimensional, difference will be the distance between states

        return reward

    def h_function(self, next_state, state, goal, goal_index):
        return state[goal_index] + goal - next_state[goal_index]

    def check_goal(self, state, goal, threshold):
        difference = abs(state[2] - goal)

        if difference > threshold:
            return False
        return True



    def run_HAC(self, env, i_level, state, tot_time, test):

        time = 0
        dt = 60

        # show_goal_achieve = True
        final_goal = self.goal
        next_state = None
        goal_concentration_reached = False
        max_goal = torch.from_numpy(np.array(self.max_goal))

        steps = 0  # number of steps taken by agent

        episode_reward_h = 0
        state_sequence, goal_sequence, action_sequence, intri_reward_sequence, reward_h_sequence = [], [], [], [], []
        goal = self.HAC[i_level].select_action_High(state)

        # goal = goal + np.random.normal(0, self.exploration_state_noise)

        goal = goal.clip(self.state_clip_low[self.goal_index], self.state_clip_high[self.goal_index])

        # for t in range(step, max_timestep):

        while time < tot_time:
            steps = steps + 1
            action = self.HAC[i_level - 1].select_action_Low(state, goal)  # action taken by lower level policy

            # action = norm_action(action)

            action = action + np.random.normal(1, self.exploration_action_noise)

            action = action.clip(self.action_clip_low, self.action_clip_high)
            # action = np.array([0.000000000000000000001])

            # 2.2.2 interact environment
            # print("state", state)
            next_state, reward = env(action, time, state)

            #next_state = np.array([next_state])
            #next_state_noise_2 = np.random.normal(next_state[:, 2], 0.01 * next_state[:, 2])
            #next_state_noise = np.concatenate((next_state[:, :2], np.array([next_state_noise_2]), next_state[:, 3:]), 1)
            #next_obs_noise = next_state_noise[0]
            next_state_noise_2 = next_state[2]


            self.lo += (np.abs(next_state_noise_2 - final_goal) ** 2)
            self.iae += (np.abs(next_state_noise_2 - final_goal))

            self.Protein.append(next_state[2])  # output variable
            self.viability.append(next_state[5])

            self.flowrate.append(action[0])

            # 2.2.3 compute step arguments
            # reward_h = self.dense_reward(state,final_goal)
            reward_h = reward

            intri_reward = self.intrinsic_reward(state, goal, next_state)
            self.reward += reward
            next_goal = self.h_function(next_state, state, goal, self.goal_index)
            # next_goal = next_goal.clip(self.state_clip_low[self.goal_index], self.state_clip_high[self.goal_index])

            # print("next goal shape",next_goal.shape)
            # print("goal", next_goal)
            # 2.2.4 collect low-level experience
            self.replay_buffer[i_level - 1].add((state, action, goal, intri_reward, next_state, next_goal, self.gamma))

            state_sequence.append(torch.from_numpy(state))
            action_sequence.append(torch.from_numpy(action))
            intri_reward_sequence.append(intri_reward)
            goal_sequence.append(goal)
            reward_h_sequence.append(reward_h)

            # update the DDPG parameters
            if len(self.replay_buffer[i_level - 1].buffer) > self.batch_size:
                self.update(i_level - 1, self.n_iter, self.batch_size)
            # experience_buffer_l.add(state, goal, action, intri_reward, next_state, next_goal, done_l)
            # 2.2.5 record segment arguments

            episode_reward_h += reward_h

            if (steps + 1) % self.c == 0 and steps > 0:

                next_goal = self.HAC[i_level].select_action_High(state)
                next_goal = next_goal + np.random.normal(0, self.exploration_state_noise)
                next_goal = next_goal.clip(self.state_clip_low[self.goal_index], self.state_clip_high[self.goal_index])

                actor_target_l =  copy.deepcopy(self.HAC[i_level - 1].actor_Low)

                # off-policy correction
                #goal_hat, updated = self.off_policy_correction(actor_target_l, action_sequence, state_sequence, self.goal_dim, self.goal_index, goal_sequence[0], next_state, max_goal, device)
                # print(goal_hat)

                # self.replay_buffer[i_level].add(state, goal_hat, episode_reward_h, next_state, done_h)
                # print("next goal", next_goal)
                # print("goal hat", goal_hat)
                self.replay_buffer[i_level].add((state_sequence[0], next_goal, episode_reward_h, next_state,
                                                 self.gamma))  # implement goal hat instead of next_goal
                state_sequence, action_sequence, intri_reward_sequence, goal_sequence, reward_h_sequence = [], [], [], [], []
                episode_reward_h = 0
                if len(self.replay_buffer[i_level].buffer) > self.batch_size:
                    self.update(i_level, self.n_iter, self.batch_size)
                # print(episode_reward_h)
                # if state_print_trigger.good2log(t, 500): print_cmd_hint(params=[state_sequence, goal_sequence, action_sequence, intri_reward_sequence, updated, goal_hat, reward_h_sequence], location='training_state')
                # 2.2.9 reset segment arguments & log (reward)

            # 2.2.10 update observations

            if next_state[2] >= final_goal and not goal_concentration_reached:
                goal_concentration_reached = True
                self.timetaken = time

            state = next_state
            goal = next_goal

            time = time + dt

            goal_achieved = self.check_goal(next_state, final_goal, self.threshold)
            # if goal_achieved and Train == True:
            # print("goal achieved in steps",t)
            # show_goal_achieve = False
            # break
            if goal_achieved and test == True:
                self.solved = True

        return next_state

    def update(self, k_level, n_iter, batch_size):
        # def update(self, n_iter, batch_size):
        # for i in range(self.k_level):
        if k_level == 0:
            self.HAC[k_level].update_Low(self.replay_buffer[k_level], n_iter, batch_size)
        else:
            self.HAC[k_level].update_High(self.replay_buffer[k_level], n_iter, batch_size)

    def save(self, directory, name):
        for i in range(self.k_level):
            self.HAC[i].save(directory, name + '_level_{}'.format(i))

    def load(self, directory, name):
        for i in range(self.k_level):
            self.HAC[i].load(directory, name + '_level_{}'.format(i))


def train():
    #################### Hyperparameters ####################
    env_name = "mAb_control"

    save_episode = 5  # keep saving every n episodes
    max_episodes = 100  # max num of training episodes



    high = np.array([5400, 9, 590, 10, 2.5, 96.5])
    # high = 590

    observation_space = spaces.Box(
        low=np.array(
            [5400, 4.147507600512498, 107.96076361017765, 2.614975072822183, 1.8767447491163762, 98.31311438674405]),
        high=high,
        dtype=np.float32
    )
    high = np.array([5], dtype=np.float32)
    # high=310
    action_space = spaces.Box(
        low=np.array([0.5]),
        high=high,
        dtype=np.float32
    )
    action_space2 = spaces.Box(
        low=np.array([5]),
        high=np.array([0.5]),
        dtype=np.float32
    )

    # dimensions of action, state, and final goal
    state_dim = observation_space.high.shape[0]
    action_dim = action_space.high.shape[0]
    goal_dim = 1  # goal is single dimension and the 3rd element of state array
    goal_index = 2

    # primitive action, goal, and state bounds and offset
    action_offset_np = action_space.low[0]
    action_bounds_np = action_space.high - action_space.low
    # goal_offset_np = np.array([0])
    action_offset = torch.FloatTensor(action_offset_np.reshape(1, -1)).to(device)
    action_bounds = torch.FloatTensor(action_bounds_np.reshape(1, -1)).to(device)

    action_clip_low = np.array(action_space.low[0])
    action_clip_high = np.array(action_space.high[0])

    # state bounds and offset
    state_offset_np = observation_space.low
    state_bounds_np = observation_space.high - observation_space.low
    state_offset = torch.FloatTensor(state_offset_np.reshape(1, -1)).to(device)
    state_bounds = torch.FloatTensor(state_bounds_np.reshape(1, -1)).to(device)
    state_clip_low = np.array(observation_space.low)
    state_clip_high = np.array(observation_space.high)
    # max_goal = observation_space.high[2]
    max_goal = np.array([600])
    # goal offset
    # goal_offset_np = np.array([0])
    # goal_offset = torch.FloatTensor(goal_offset_np.reshape(1, -1)).to(device)
    # exploration noise std for primitive action and subgoals
    exploration_action_noise = np.array([0.1])
    exploration_state_noise = np.array([0.1])
    action_policy_noise = np.array([0.2])
    state_policy_noise = np.array([0.2])
    action_policy_clip = np.array([0.5])
    state_policy_clip = np.array([0.5])

    goal = np.array([590])  # final goal state to be achived
    threshold = np.array([1])  # threshold value to check if goal state is achieved

    # HAC parameters:
    k_level = 2  # num of levels in hierarchy
    c = 10  # time horizon to achieve subgoal
    lamda = 0.3  # subgoal testing parameter

    # DDPG parameters:
    gamma = 0.95  # discount factor for future rewards
    # n_iter = 100                # update policy n_iter times in one DDPG update
    # changing the n_iter from 100 to 6
    n_iter = 6
    # batch_size = 100            # num of transitions sampled from replay buffer
    # changing batch size from 100 to 5
    batch_size = 5
    lr = 0.001
    policy_freq = 2  # policy frequency to update TD3
    tau = 0.005

    # save trained models
    directory = "./preTrained/{}/{}level/".format(env_name, k_level)
    directory_HIRO = "./Reward_Plots/"
    directory_HIRO_plot_G = "./Plot_G/"
    filename = "HAC_{}".format(env_name)
    #########################################################

    # creating HAC agent and setting parameters
    # seed = 50
    # torch.manual_seed(seed)

    agent = HAC(k_level, policy_freq, tau, c, state_dim, action_dim, goal_dim, goal_index, goal, render, threshold,
                action_offset, state_offset, action_bounds, state_bounds, max_goal, action_policy_noise,
                state_policy_noise, action_policy_clip, state_policy_clip, lr)
    agent.set_parameters(lamda, gamma, n_iter, batch_size, action_clip_low, action_clip_high,
                         state_clip_low, state_clip_high, exploration_action_noise, exploration_state_noise)

    # logging file:
    log_f = open("log.txt", "w+")

    # training procedure
    # agent.success_rate = np.zeros(max_episodes)
    agent.episode_rewards = []
    agent.rmse = []
    agent.IAE = []
    agent.average_reward = []
    agent.average_rmse = []
    agent.average_iae = []

    agent.Protein_TD3_reward = []

    # success = np.zeros(max_episodes)
    # successful = 0

    # set is a train case
    Test = False

    x0 = np.asarray([5400, 4.147507600512498, 107.96076361017765, 2.614975072822183, 1.8767447491163762,
          98.31311438674405])


    for i_episode in range(1, max_episodes + 1):
        agent.timetaken = 0  # time taken by the system to reach the goal concentration
        agent.reward = 0
        agent.lo = 0  # rmse
        agent.iae = 0
        agent.Protein = []
        agent.viability = []
        agent.flowrate = []

        # agent.success = []
        agent.timestep = 0
        tot_time = 15*24*60
        state = x0  # initial state
        # collecting experience in environment
        last_state = agent.run_HAC(get_state, k_level - 1, state, tot_time, Test)
        # print("last sate",last_state[3])
        if agent.check_goal(last_state, goal, threshold):
            print("################ Solved! ################ ")
            # successful = successful + 1
            name = filename + '_solved'
            agent.save(directory, name)

        # update all levels
        # print("lo",agent.lo)
        # agent.update(n_iter, batch_size)
        agent.episode_rewards.append(agent.reward)
        agent.rmse.append(math.sqrt(agent.lo / 360))
        agent.IAE.append(agent.iae)
        agent.Protein_TD3_reward.append(agent.Protein)

        agent.average_reward.append(np.mean(agent.episode_rewards[-10:]))
        agent.average_rmse.append(np.mean(agent.rmse[-10:]))
        agent.average_iae.append(np.mean(agent.IAE[-10:]))

        # logging updates:
        log_f.write('{},{}\n'.format(i_episode, agent.reward))
        log_f.flush()
        # print("Last State",last_state[3])
        if i_episode % save_episode == 0:
            agent.save(directory, filename)


        print("Episode: {}\t Reward: {}".format(i_episode, agent.reward))
        print("state: ", last_state[2])
        print("Time taken: ", agent.timetaken)
        print("RMSE:", agent.rmse[i_episode - 1])

        name = directory_HIRO_plot_G + str(i_episode)
        plot_G(agent.Protein, tot_time, agent.flowrate, name)

    font1 = {'family': 'serif', 'size': 15}
    font2 = {'family': 'serif', 'size': 15}

    plt.figure()
    plt.plot(agent.episode_rewards)
    plt.xlabel("Number of episodes", fontdict=font2)
    plt.ylabel("Rewards", fontdict=font2)
    plt.savefig(directory_HIRO + 'Reward_Per_Episode_HIRO.png', bbox_inches='tight')
    plt.close()
    # plt.show()

    plt.figure()
    plt.plot(agent.average_reward)
    plt.xlabel("Number of episodes", fontdict=font1)
    plt.ylabel("Average Rewards", fontdict=font2)
    plt.savefig(directory_HIRO + 'Average_Rewarde_HIRO.png', bbox_inches='tight')
    plt.close()

    plt.figure()
    plt.plot(agent.average_rmse)
    plt.xlabel("Number of episodes", fontdict=font2)
    plt.ylabel("Average RMSE", fontdict=font2)
    plt.savefig(directory_HIRO + 'RMSE_HIRO.png', bbox_inches='tight')
    plt.close()

    plt.figure()
    plt.plot(agent.average_iae)
    plt.xlabel("Number of episodes", fontdict=font2)
    plt.ylabel("Average IAE", fontdict=font2)
    plt.savefig(directory_HIRO + 'IAE_HIRO.png', bbox_inches='tight')
    plt.close()

    np.savetxt("Protein_TD3_reward.csv", agent.Protein_TD3_reward, delimiter=",")


if __name__ == '__main__':
    train()

