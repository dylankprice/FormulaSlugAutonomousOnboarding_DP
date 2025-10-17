import matplotlib.pyplot as plt
import matplotlib.animation as animation
from dataclasses import dataclass
import numpy as np


xpoints = np.array([0, 6])
ypoints = np.array([0, 250])

plt.plot(xpoints, ypoints)
plt.show()

@dataclass
class State:
    pass

deltaT = .5

def step (state:State) -> State:
    return state

def plot():
    pass

fig = plt.figure(figsize=(3,3), dpi=150)
ax = fig.add_subplot(111)
ax.grid()
ax.set_xlim(-2, 2)
ax.set_ylim(-2, 2)
plt.pause(5)
ani = animation.FuncAnimation(fig, plot, interval=0)
plt.show()