import matplotlib.pyplot as plt
import matplotlib.animation as animation
from dataclasses import dataclass
import numpy as np

@dataclass
class State:
    xpos: float
    ypos: float
    xvel: float
    yvel: float

deltaT = .5
bounceCoeff = 1.1

def step (state:State) -> State:
    new_xpos = state.xpos + state.xvel * deltaT
    new_ypos = state.ypos + state.yvel * deltaT
    new_yvel = state.yvel - 9.81 * deltaT
    new_xvel = state.xvel - 0 * deltaT
    if new_ypos < 3:
        new_yvel = new_yvel * -1 * bounceCoeff
    if new_ypos > 1000:
        new_yvel = new_yvel * -1 * bounceCoeff
    if new_xpos < 0:
        new_xvel = new_xvel * -1 * bounceCoeff
    if new_xpos > 100:
        new_xvel = new_xvel * -1 * bounceCoeff
    #ball collision code
    #if abs(s0.xpos - s1.xpos) <= 3 and abs(s0.ypos - s1.ypos) <= 3:
        #new_xvel = new_xvel * -1 * bounceCoeff
        #new_yvel = new_yvel * -1 * bounceCoeff
    sNew = State(
        xpos=new_xpos,
        ypos=new_ypos,
        xvel=new_xvel,
        yvel=new_yvel
    )
    return sNew

def animate (i):
    global s0
    global s1
    s0 = step(s0)
    s1 = step(s1)
    ax.clear()
    ax.scatter([s0.xpos], [s0.ypos], s = 200)
    ax.scatter([s1.xpos], [s1.ypos], s = 200)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 1000)
    return ax,

s0 = State(
    xpos=50,
    ypos=500,
    xvel=4,
    yvel=0
)

s1 = State(
    xpos=100,
    ypos=500,
    xvel=-4,
    yvel=0
)

fig = plt.figure(figsize=(3,3), dpi=150)
ax = fig.add_subplot(111)
ax.grid()
ax.set_xlim(-2, 2)
ax.set_ylim(-2, 2)
plt.pause(5)
ani = animation.FuncAnimation(fig, animate, interval=0)
plt.show()