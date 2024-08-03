import numpy as np
import gym
from gym import spaces
tanh = np.tanh
import matplotlib.pyplot as plt
import math
import torch
pow = math.pow
exp = np.exp
import itertools
import time
import random
from math import exp as exp

# In[2]:


from math import exp
import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt

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

seed = 12368
torch.manual_seed(seed)
high = np.array([5400, 9, 590 , 10, 2.5, 96.5])
#high = 590
observation_space = spaces.Box(
            low=np.array([5400, 4.147507600512498, 107.96076361017765 , 2.614975072822183, 1.8767447491163762, 98.31311438674405]),
            high=high,
            dtype=np.float32
        )
high = np.array([5], dtype=np.float32)
#high=310
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

#high = np.array([5,25,310.15], dtype=np.float32)
#action_space = spaces.Box(
            #low=np.array([0.5,1,308]),
            #high=high,
            #dtype=np.float32
