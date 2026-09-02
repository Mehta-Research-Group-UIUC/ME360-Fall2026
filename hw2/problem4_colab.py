import numpy as np
import matplotlib.pyplot as plt


FONTSIZE = 16
plt.rc('font', size=FONTSIZE)          # controls default text sizes

def ax_setting(ax: plt.Axes, time_horizon: float, sampling_rate: float) -> plt.Axes:
	ax.set_title(f"Sampling Rate: {sampling_rate:.2f} Hz")
	ax.annotate('', xy=(1.1*time_horizon, 0), xytext=(-0.1*time_horizon, 0),
				arrowprops=dict(arrowstyle="->", color='black', linewidth=2))
	ax.annotate('', xy=(0, 1.2), xytext=(0, -1.2),
				arrowprops=dict(arrowstyle="->", color='black', linewidth=2))
	ax.spines['top'].set_visible(False)
	ax.spines['right'].set_visible(False)
	ax.spines['left'].set_position('zero')
	ax.spines['bottom'].set_position('zero')
	ax.spines['left'].set_visible(False)
	ax.spines['bottom'].set_visible(False)
	ax.yaxis.tick_left()
	ax.xaxis.tick_bottom()
	ax.set_xlim([-0.1*time_horizon, 1.1*time_horizon])
	ax.set_ylim([-1.2, 1.2])
	ax.set_xlabel("time [sec]")
	ax.xaxis.set_label_coords(1.04, 0.475)
	ax.set_ylabel("signal value", loc='top', rotation=0)
	ax.yaxis.set_label_coords(0.15, 1.04)
	ax.tick_params(axis='both', width=2)
	ax.tick_params(axis='both', length=8)
	ax.set_xticks([tick_value for tick_value in ax.get_xticks() if tick_value > 0 and tick_value <= time_horizon])
	ax.set_yticks([tick_value for tick_value in ax.get_yticks() if tick_value != 0 and abs(tick_value) <= 1])
	ax.plot(0, 0, 'ko')  # plot a black dot at the origin
	ax.text(-0.05*time_horizon, -0.1, '0.0', fontsize=FONTSIZE)  # add text '0' at the bottom left corner of the origin
	return ax

def sinusoidal_signal(time):
	frequency = 30. # frequency of 30Hz
	return np.sin(2*np.pi*frequency*time)

time_horizon = 0.55 # time horizon of 0.55 seconds
number_of_points = 10001 # number of points for continuous signal plot
continuous_time = np.linspace(0, time_horizon, number_of_points, endpoint=True) 
continuous_signal =  sinusoidal_signal(continuous_time)

########## Change the sampling frequency ##########

sampling_rate = 100. # sampling frequency of 100 Hz

###################################################

sampled_time = np.arange(0, time_horizon, 1./sampling_rate)
sampled_signal = sinusoidal_signal(sampled_time)

fig, ax = plt.subplots(1,1, figsize=(12, 9))
ax = ax_setting(ax, time_horizon, sampling_rate)
ax.plot(continuous_time, continuous_signal, alpha=0.5, label="continuous signal", linewidth=5)
ax.plot(sampled_time, sampled_signal, label="sampled signal", color="C1", linewidth=5)
ax.legend(loc='upper center', bbox_to_anchor=(0.5, 0.), ncol=2)
plt.show()
