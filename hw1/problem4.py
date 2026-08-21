# !sudo apt-get install texlive-full
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams.update({"text.usetex": True})

FONTSIZE = 16

def signal(time):
    y = np.zeros(time.shape)
    for i, t in enumerate(time):
        if t >= 0:
            y[i] = (np.sin(t))**3
    return y

t = np.linspace(-3, 22, 1000)
y = signal(t)

fig, ax = plt.subplots(1,1,figsize=(10, 6))
fig.suptitle(
    r'Plot of $\sin^3(t)$ for $t > 0$, and $0$ otherwise', 
    fontsize=FONTSIZE+2
)
ax.set_xlabel(r'$t$', fontsize=FONTSIZE)
ax.set_ylabel(r'$y$', fontsize=FONTSIZE)
ax.grid(True)
ax.axhline(y=0, color='black', linestyle='-', alpha=1)
ax.axvline(x=0, color='black', linestyle='-', alpha=1)
ax.plot(t, y, linewidth=5)
ax.set_xlim(-2, 21)
ax.tick_params(axis='both', which='major', labelsize=FONTSIZE)
plt.show()
