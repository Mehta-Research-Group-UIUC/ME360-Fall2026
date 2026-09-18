import numpy as np
import matplotlib.pyplot as plt

FONTSIZE = 16

def signal1(time):
    y = np.zeros(time.shape)
    for i, t in enumerate(time):
        if t >= 1.0:
            y[i] = 1.0
    return y

def signal2(time):
    pass

def signal3(time):
    pass

def signal4(time):
    pass

def signal5(time):
    pass


def signal(time,probno):
	if probno == 1:
		return signal1(time)
	elif probno == 2:
		return signal2(time)
	elif probno == 3:
		return signal3(time)
	elif probno == 4:
		return signal4(time)
	elif probno == 5:
		return signal5(time)


t = np.linspace(-3, 10, 1000)

f1, axs = plt.subplots(5,1,figsize=(6, 9))

for i in range(1,6):
	ax = axs[i-1]
	y = signal(t,i)
	ax.plot(t, y, linewidth=5,label=str(i))
	ax.legend(framealpha=1.0)
	ax.set_xlabel(r'$t$', fontsize=FONTSIZE)
	ax.set_ylabel(r'$y$', fontsize=FONTSIZE)
	ax.grid(True)
	ax.axhline(y=0, color='black', linestyle='-', alpha=1)
	ax.axvline(x=0, color='black', linestyle='-', alpha=1)
	ax.set_xlim(-3, 10)
	ax.tick_params(axis='both', which='major', labelsize=FONTSIZE)

f1.suptitle(
    r'Plots for Q4', 
    fontsize=FONTSIZE+2
)


plt.savefig("problem4_soln.png")
plt.show()
