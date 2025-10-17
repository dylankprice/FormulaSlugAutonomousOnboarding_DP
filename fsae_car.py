import time
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from dataclasses import dataclass
import numpy as np

@dataclass
#Model a car accelerating at 2 m/s for 10 sec and then braking at 4 m/s till stopping. 
#Implement Drag in your car model Cross Sectional Area * drag coeff = 1.1 Graphed with Matplotlib
class State:
    xpos: float
    ypos: float
    xvel: float
    yvel: float
    time: float

deltaT = 0.01
accel = 0
drag_Coeff = 1.1

def step (state:State) -> State:
    if state.time < 10:
        accel = 2
    elif state.time >= 10 and state.xvel > 0:
        accel = -4

    new_xpos = state.xpos + state.xvel * deltaT
    new_xvel = state.xvel + accel * state.time

    sNew = State(
        xpos=new_xpos,
        ypos=state.ypos,
        xvel=new_xvel,   
        yvel=state.yvel,
        time=state.time + deltaT
    )
    return sNew




def animate (i):
    global s0
    s0 = step(s0)
    ax.clear()
    ax.scatter([s0.xpos], [s0.ypos], s = 200)
    ax.set_xlim(0, 1000)
    ax.set_ylim(0, 10)
    return ax,

s0 = State(
    xpos=10,
    ypos=0,
    xvel=0,
    yvel=0,
    time=0
)

fig = plt.figure(figsize=(3,3), dpi=150)
ax = fig.add_subplot(111)
ax.grid()
ax.set_xlim(0, 1000)
ax.set_ylim(0, 10)
plt.pause(5)
ani = animation.FuncAnimation(fig, animate, interval=0)
plt.show()