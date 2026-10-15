import numpy as np
import matplotlib.pyplot as plt

number_of_time_steps = 16

y = np.zeros(number_of_time_steps+1)
u = np.zeros(number_of_time_steps)

y[-1] = 0.0 # initial condition

for n in range(number_of_time_steps):
	u[n] = 1.0 # This is input
	y[n] = y[n-1] + u[n] # This is system update

fig, ax = plt.subplots(1,1)
ax.stem(np.arange(number_of_time_steps), y[:-1])
ax.set_xlabel("time step")
ax.set_ylabel("y")

fig.suptitle("Problem 4 demo")
plt.show()
