import numpy as np
tanh = np.tanh
import math
pow = math.pow
exp = np.exp
import sys
from gym import spaces
import torch.nn.functional as F
import torch.autograd
from torch.autograd import Variable
import torch
import torch.autograd
import torch.optim as optim
import torch.nn as nn
from math import exp
from scipy.integrate import odeint
import matplotlib.pyplot as plt
import numpy as np
from collections import deque
import random



seed = 12368
random.seed(seed)
torch.manual_seed(seed)
np.random.seed(seed)

def get_state(action, dt, ti, x0):
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


# In[3]:


x0 = [5400, 4.147507600512498, 107.96076361017765, 2.614975072822183, 1.8767447491163762, 98.31311438674405]

# In[4]:





class Memory:
    def __init__(self, max_size):
        self.max_size = max_size
        self.buffer = deque(maxlen=max_size)

    def push(self, state, action, reward, next_state):
        experience = (state, action, reward, next_state)
        self.buffer.append(experience)

    def sample(self, batch_size):
        state_batch = []
        action_batch = []
        reward_batch = []
        next_state_batch = []

        batch = random.sample(self.buffer, batch_size)
        for experience in batch:
            state, action, reward, next_state = experience
            state_batch.append(state)
            action_batch.append(action)
            reward_batch.append(reward)
            next_state_batch.append(next_state)

        return state_batch, action_batch, reward_batch, next_state_batch

    def __len__(self):
        return len(self.buffer)


# In[5]:




device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Actor(nn.Module):
    def __init__(self, hidden_size, learning_rate=3e-4):
        super(Actor, self).__init__()
        self.linear1 = nn.Linear(6, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.linear3 = nn.Linear(hidden_size, 1)

    def forward(self, state):
        x = F.relu(self.linear1(state))
        x = F.relu(self.linear2(x))
        x = F.relu(self.linear3(x))
        #         x = self.linear3(x)
        return x


class Critic(nn.Module):
    def __init__(self, hidden_size):
        super(Critic, self).__init__()
        # Q1
        self.linear1 = nn.Linear(7, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.linear3 = nn.Linear(hidden_size, 1)
        # Q2
        self.linear4 = nn.Linear(7, hidden_size)
        self.linear5 = nn.Linear(hidden_size, hidden_size)
        self.linear6 = nn.Linear(hidden_size, 1)

    def forward(self, state, action):
        x = torch.cat([state.reshape((6, 6)), action.reshape((6, 1))], dim=1)
        x1 = F.relu(self.linear1(x))
        x1 = F.relu(self.linear2(x1))
        x1 = self.linear3(x1)
        x2 = F.relu(self.linear4(x))
        x2 = F.relu(self.linear5(x2))
        x2 = self.linear6(x2)
        return x1, x2

    def get_Q(self, state, action):
        x = torch.cat([state.reshape((6, 6)), action.reshape((6, 1))], dim=1)
        x1 = F.relu(self.linear1(x))
        x1 = F.relu(self.linear2(x1))
        x1 = self.linear3(x1)
        return x1


# In[6]:





class TD3(object):
    def __init__(self, action, states, max_action, min_action, num_actions, hidden_size=256, actor_learning_rate=1e-4,
                 critic_learning_rate=1e-3, gamma=0.99, tau=1e-2, policy_freq=2, policy_noise=0.2, noise_clip=0.5,
                 max_memory_size=50000):
        self.num_states = states
        self.num_actions = action
        self.gamma = gamma
        self.tau = tau
        self.policy_freq = policy_freq
        self.policy_noise = policy_noise
        self.noise_clip = noise_clip
        self.max_action = max_action
        self.min_action = min_action

        self.actor = Actor(hidden_size).to(device)
        self.actor_target = Actor(hidden_size).to(device)

        self.critic = Critic(hidden_size).to(device)
        self.critic_target = Critic(hidden_size).to(device)

        self.memory = Memory(max_memory_size)
        self.critic_criterion = nn.MSELoss()
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=actor_learning_rate)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=critic_learning_rate)

    def get_action(self, state, noise=0.1):
        state = Variable(torch.from_numpy(state).float().unsqueeze(0)).to(device)
        action = self.actor.forward(state).to(device)
        action = action.cpu().detach().numpy()
        action = (action + np.random.normal(1, noise, size=self.num_actions))
        action.clip(self.min_action, self.max_action)
        return action.clip(self.min_action, self.max_action)

    def train(self, iterations, batch_size, discount=0.99, tau=0.005, policy_noise=0.2, noise_clip=0.5, policy_freq=4):
        states, actions, rewards, next_states = self.memory.sample(batch_size)
        states = torch.transpose(Variable(torch.from_numpy(np.array(states)).float().unsqueeze(0)), 0, 1).to(device)
        actions = (Variable(torch.from_numpy(np.array(actions)).float())).to(device)
        rewards = Variable(torch.from_numpy(np.array(rewards)).float().unsqueeze(1)).to(device)
        next_states = torch.transpose(Variable(torch.from_numpy(np.array(next_states)).float().unsqueeze(0)), 0, 1).to(
            device)

        for it in range(iterations):
            noise = actions.data.normal_(0, policy_noise).to(device)
            noise = noise.clamp(-noise_clip, noise_clip)
            next_actions = self.actor_target.forward(next_states)
            next_actions = (next_actions + noise).clamp(self.min_action, self.max_action)
            target_Q1, target_Q2 = self.critic_target.forward(next_states, next_actions)
            target_Q = torch.min(target_Q1, target_Q2)
            target_Q = rewards + (discount * target_Q).detach()
            current_Q1, current_Q2 = self.critic.forward(states, actions)
            critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)
            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            self.critic_optimizer.step()
            if it % policy_freq == 0:
                actor_loss = -self.critic.get_Q(states, self.actor(states)).mean()
                self.actor_optimizer.zero_grad()
                actor_loss.backward()
                self.actor_optimizer.step()
                for target_param, param in zip(self.actor_target.parameters(), self.actor.parameters()):
                    target_param.data.copy_(param.data * self.tau + target_param.data * (1.0 - self.tau))
                for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
                    target_param.data.copy_(param.data * self.tau + target_param.data * (1.0 - self.tau))


# In[7]:


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


# In[8]:




# seed=50


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

# high = np.array([5,25,310.15], dtype=np.float32)
# action_space = spaces.Box(
# low=np.array([0.5,1,308]),
# high=high,
# dtype=np.float32

agent = TD3(action_space.shape[0], observation_space.shape[0], max_action=action_space.high[0],
            min_action=action_space.low[0], num_actions=action_space.shape[0])






batch_size = 6
tot_time = 15 * 24 * 60
dt = 60
rewards = []
avg_rewards = []
rmse = []
IAE = []
avg_rmse = []
avg_IAE = []
loss_p = []
loss_c = []
episode_reward = []
Protein_TD3_reward = []
Least_Time = sys.maxsize  # minimum time in which goal concentration is achieved
Least_Time_Episode = 0 # episode in which least time is achieved


directory_TD3 = "./Reward_Plots/"
directory_TD3_plot_G = "./Plot_G/"

for episode in range(100):
    x0 = [5400, 4.147507600512498, 107.96076361017765, 2.614975072822183, 1.8767447491163762,
          98.31311438674405]  # 107.96076361017765
    time_taken = 0 # time taken by the system to reach the goal concentration
    goal_concentration_reached = False

    t = 0
    Protein = []
    viability = []
    flowrate = []
    episode_reward = 0
    total_reward = 0
    lo = 0  # rmse
    iae = 0
    while t < tot_time:
        Protein.append(x0[2])  # output variable
        viability.append(x0[5])
        state = x0  # concentration
        action = agent.get_action(np.array(state))

        flowrate.append(action[0])
        new_state, reward = get_state(action[0], dt, t, x0)
        agent.memory.push(x0, action, reward, new_state)
        if len(agent.memory) > batch_size:
            agent.train(6, batch_size)
        t = t + dt
        x0 = new_state
        lo += np.abs(new_state[2] - 590) ** 2
        iae += np.abs(new_state[2] - 590)
        reward = np.array(reward).flatten()
        episode_reward += reward
        if x0[2] >= 590 and not goal_concentration_reached:
            goal_concentration_reached = True
            time_taken = t


    name = directory_TD3_plot_G + str(episode + 1)
    plot_G(Protein, tot_time,flowrate, name)
    lo = math.sqrt(lo / 360)
    iae = iae

    if time_taken < Least_Time:
        Least_Time = time_taken
        Least_Time_Episode = episode + 1

    rmse.append(lo)
    IAE.append(iae)
    rewards.append(episode_reward[0])
    avg_rewards.append(np.mean(rewards[-10:]))
    avg_rmse.append(np.mean(rmse[-10:]))
    avg_IAE.append(np.mean(IAE[-10:]))
    Protein_TD3_reward.append(Protein)

    print("batch: ", episode + 1, " reward: ", episode_reward)
    print("last state", x0[2])
    print("Time taken: ", time_taken)
    print("RMSE:", rmse[episode])

print("Least Time", Least_Time)
print("Least Time Episode", Least_Time_Episode)

np.savetxt("Protein_TD3_reward.csv", Protein_TD3_reward, delimiter=",")



font1 = {'family': 'serif', 'size': 15}
font2 = {'family': 'serif', 'size': 15}



plt.figure()
plt.plot(rewards)
plt.xlabel("Number of episodes", fontdict=font2)
plt.ylabel("Rewards", fontdict=font2)
plt.savefig(directory_TD3+'Reward_Per_Episode_TD3.png', bbox_inches = 'tight')
plt.close()

#plt.show()

plt.figure()
plt.plot(avg_rewards)
plt.xlabel("Number of episodes", fontdict=font1)
plt.ylabel("Average Rewards", fontdict=font2)
plt.savefig(directory_TD3+'Average_Rewarde_TD3.png', bbox_inches = 'tight')
plt.close()

#plt.show()
# In[ ]:
plt.figure()
plt.plot(avg_rmse)
plt.xlabel("Number of episodes", fontdict=font2)
plt.ylabel("Average RMSE", fontdict=font2)
plt.savefig(directory_TD3+'RMSE_TD3.png', bbox_inches = 'tight')
plt.close()

#plt.show()

plt.figure()
plt.plot(avg_IAE)
plt.xlabel("Number of episodes", fontdict=font2)
plt.ylabel("Average IAE", fontdict=font2)
plt.savefig(directory_TD3+'IAE_TD3.png', bbox_inches = 'tight')
plt.close()





