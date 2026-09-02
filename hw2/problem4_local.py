import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
import tkinter as tk
from tkinter import ttk

FONTSIZE = 10
plt.rc('font', size=FONTSIZE)

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
    ax.set_xlabel("time\n[sec]")
    ax.xaxis.set_label_coords(1.04, 0.475)
    ax.set_ylabel("signal value")
    ax.tick_params(axis='both', width=1)
    ax.tick_params(axis='both', length=4)
    ax.set_xticks([tick_value for tick_value in ax.get_xticks() if tick_value > 0 and tick_value <= time_horizon])
    ax.set_yticks([tick_value for tick_value in ax.get_yticks() if tick_value != 0 and abs(tick_value) <= 1])
    ax.plot(0, 0, 'ko')
    ax.text(-0.1*time_horizon, -0.15, '0.0', fontsize=FONTSIZE)
    return ax

class SignalSamplingApp:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("Signal Sampling Visualization")


        self.time_horizon = 0.55
        self.number_of_points = 10001
        self.continuous_time = np.linspace(0, self.time_horizon, self.number_of_points, endpoint=True)
        self.continuous_signal = sinusoidal_signal(self.continuous_time)

        self.fig, self.ax = plt.subplots(figsize=(5, 3))
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.root)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)

        self.toolbar = NavigationToolbar2Tk(self.canvas, self.root)
        self.toolbar.update()
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)

        self.default_sampling_rate = 100.0
        self.minimum_smapling_rate = 1.0
        self.maximum_smapling_rate = 200.0
        self.sampling_rate = tk.DoubleVar(value=self.default_sampling_rate)
        self.create_widgets()

        self.update_plot()

    def create_widgets(self):
        frame = ttk.Frame(self.root)
        frame.pack(side=tk.BOTTOM, fill=tk.X)

        ttk.Label(frame, text="Sampling Rate (Hz):").pack(side=tk.LEFT, padx=(10, 0))
        sampling_slider = ttk.Scale(frame, from_=self.minimum_smapling_rate, to=self.maximum_smapling_rate, orient=tk.HORIZONTAL, 
                                    variable=self.sampling_rate, command=self.on_slider_change)
        sampling_slider.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=10)
        
        sampling_entry = ttk.Entry(frame, textvariable=self.sampling_rate, width=10)
        sampling_entry.pack(side=tk.LEFT, padx=(0, 10))
        sampling_entry.bind('<Return>', self.on_entry_change)

    def on_slider_change(self, event):
        self.update_plot()

    def on_entry_change(self, event):
        try:
            value = float(self.sampling_rate.get())
            if self.minimum_smapling_rate <= value <= self.maximum_smapling_rate:
                self.update_plot()
            else:
                self.sampling_rate.set(self.default_sampling_rate)
        except ValueError:
            self.sampling_rate.set(self.default_sampling_rate)

    def update_plot(self):
        self.ax.clear()
        sampling_rate = self.sampling_rate.get()
        sampled_time = np.arange(0, self.time_horizon, 1./sampling_rate)
        sampled_signal = sinusoidal_signal(sampled_time)

        ax_setting(self.ax, self.time_horizon, sampling_rate)
        

        self.ax.plot(self.continuous_time, self.continuous_signal, alpha=0.5, label="30 [Hz] continuous signal", linewidth=3)
        self.ax.plot(sampled_time, sampled_signal, label="sampled signal", color="C1", linewidth=3)

        self.ax.legend(loc='upper center', bbox_to_anchor=(0.5, 0.), ncol=2)
        self.canvas.draw()

def sinusoidal_signal(time: np.ndarray) -> np.ndarray:
    frequency = 30.  # frequency of 30Hz
    return np.sin(2*np.pi*frequency*time)

if __name__ == "__main__":
    root = tk.Tk()
    app = SignalSamplingApp(root)
    root.mainloop()
