import tkinter as tk
from tkinter import ttk, messagebox
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from scipy import signal
import matplotlib.patches as patches

class FilterDesignTool:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("Design Filter")
        
        self.control_frame = ttk.Frame(root)
        self.control_frame.pack(side=tk.LEFT, fill=tk.Y, padx=5, pady=5)
        
        self.plot_frame = ttk.Frame(root)
        self.plot_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        self._init_controls()
        self._init_plots()
        self.design_filter()  # Initial plot
        
    def _init_controls(self) -> None:
        
        # Add code display frame
        self.code_frame = ttk.LabelFrame(self.control_frame, text="Python Code")
        self.code_frame.pack(fill=tk.X, pady=10, padx=5)
        
        self.code_text = tk.Text(self.code_frame, height=8, width=50, font=('Courier', 9))
        self.code_text.pack(fill=tk.X, padx=5, pady=5)

        # Add copy button
        self.copy_button = ttk.Button(self.code_frame, text="Copy Code", 
                                    command=self.copy_code)
        self.copy_button.pack(pady=(0, 5))

        # Add separator
        ttk.Separator(self.control_frame, orient='horizontal').pack(fill='x', pady=10)

        # Design Button
        self.design_button = ttk.Button(self.control_frame, text="Design Filter",
                                      command=self.design_filter)
        self.design_button.pack(pady=10)

        # Sampling Frequency
        ttk.Label(self.control_frame, text="Sampling Frequency (Hz):").pack(anchor=tk.W)
        self.fs_var = tk.StringVar(value="8000")
        self.fs_entry = ttk.Entry(self.control_frame)
        self.fs_entry.insert(0, "8000")
        self.fs_entry.pack(fill=tk.X, pady=(0, 10))
        
        # Filter Order
        ttk.Label(self.control_frame, text="Filter Order:").pack(anchor=tk.W)
        self.order_var = tk.StringVar(value="4")
        self.order_entry = ttk.Entry(self.control_frame)
        self.order_entry.insert(0, "4")
        self.order_entry.pack(fill=tk.X, pady=(0, 10))
        
        # Filter Type
        ttk.Label(self.control_frame, text="Filter Type").pack(anchor=tk.W)
        self.filter_type_var = tk.StringVar(value="lowpass")
        
        # Low Pass
        self.lowpass_frame = ttk.Frame(self.control_frame)
        self.lowpass_frame.pack(fill=tk.X, pady=2)
        ttk.Radiobutton(self.lowpass_frame, text="Lowpass", variable=self.filter_type_var, 
                       value="lowpass", command=self.update_cutoff_fields).pack(anchor=tk.W)
        
        # High Pass
        self.highpass_frame = ttk.Frame(self.control_frame)
        self.highpass_frame.pack(fill=tk.X, pady=2)
        ttk.Radiobutton(self.highpass_frame, text="Highpass", variable=self.filter_type_var, 
                       value="highpass", command=self.update_cutoff_fields).pack(anchor=tk.W)
        
        # Band Pass
        self.bandpass_frame = ttk.Frame(self.control_frame)
        self.bandpass_frame.pack(fill=tk.X, pady=2)
        ttk.Radiobutton(self.bandpass_frame, text="Bandpass", variable=self.filter_type_var, 
                       value="bandpass", command=self.update_cutoff_fields).pack(anchor=tk.W)
        
        # Band Stop
        self.bandstop_frame = ttk.Frame(self.control_frame)
        self.bandstop_frame.pack(fill=tk.X, pady=2)
        ttk.Radiobutton(self.bandstop_frame, text="Bandstop", variable=self.filter_type_var, 
                       value="bandstop", command=self.update_cutoff_fields).pack(anchor=tk.W)
        
        # Cutoff Frequencies Frame
        self.cutoff_frame = ttk.Frame(self.control_frame)
        self.cutoff_frame.pack(fill=tk.X, pady=(10, 0))
        
        # Single Cutoff (for low/high pass)
        self.single_cutoff_frame = ttk.Frame(self.cutoff_frame)
        ttk.Label(self.single_cutoff_frame, text="Cutoff Frequency (Hz)").pack(anchor=tk.W)
        self.cutoff_entry = ttk.Entry(self.single_cutoff_frame)
        self.cutoff_entry.insert(0, "1000")  # Default value
        self.cutoff_entry.pack(fill=tk.X)
        
        # Double Cutoff (for band pass/stop)
        self.double_cutoff_frame = ttk.Frame(self.cutoff_frame)
        ttk.Label(self.double_cutoff_frame, text="Low Cutoff Frequency (Hz)").pack(anchor=tk.W)
        self.low_cutoff_entry = ttk.Entry(self.double_cutoff_frame)
        self.low_cutoff_entry.insert(0, "1000")  # Default value
        self.low_cutoff_entry.pack(fill=tk.X)
        ttk.Label(self.double_cutoff_frame, text="High Cutoff Frequency (Hz)").pack(anchor=tk.W)
        self.high_cutoff_entry = ttk.Entry(self.double_cutoff_frame)
        self.high_cutoff_entry.insert(0, "2000")  # Default value
        self.high_cutoff_entry.pack(fill=tk.X)
        
        # Filter Design Method
        ttk.Label(self.control_frame, text="Design Method:").pack(anchor=tk.W, pady=(10, 0))
        self.design_method_var = tk.StringVar(value="butter")
        methods = [
            ("Butterworth", "butter"),
            ("Chebyshev Type 1", "cheby1"),
            ("Chebyshev Type 2", "cheby2"),
            ("Elliptic", "ellip"),
            ("Maximally Flat", "bessel")
        ]
        for text, value in methods:
            ttk.Radiobutton(self.control_frame, text=text, 
                          variable=self.design_method_var, 
                          value=value).pack(anchor=tk.W)
        
        # Ripple settings frame
        self.param_frame = ttk.LabelFrame(self.control_frame, text="Filter Parameters")
        self.param_frame.pack(fill=tk.X, pady=10)
        
        # Passband ripple (for Chebyshev Type 1 and Elliptic)
        self.rp_frame = ttk.Frame(self.param_frame)
        ttk.Label(self.rp_frame, text="Passband Ripple (dB):").pack(anchor=tk.W)
        self.rp_var = tk.StringVar(value="3")
        self.rp_entry = ttk.Entry(self.rp_frame, textvariable=self.rp_var)
        self.rp_entry.pack(fill=tk.X)
        
        # Stopband attenuation (for Chebyshev Type 2 and Elliptic)
        self.rs_frame = ttk.Frame(self.param_frame)
        ttk.Label(self.rs_frame, text="Stopband Attenuation (dB):").pack(anchor=tk.W)
        self.rs_var = tk.StringVar(value="40")
        self.rs_entry = ttk.Entry(self.rs_frame, textvariable=self.rs_var)
        self.rs_entry.pack(fill=tk.X)
        
        # Add trace to design method to show/hide parameters
        self.design_method_var.trace_add('write', self.update_parameter_visibility)
        
        # Show initial cutoff fields
        self.update_cutoff_fields()

    def copy_code(self):
        code = self.code_text.get(1.0, tk.END)
        self.root.clipboard_clear()
        self.root.clipboard_append(code)
        
        # Visual feedback
        self.copy_button.configure(text="Copied!")
        self.root.after(1000, lambda: self.copy_button.configure(text="Copy Code"))

    def update_parameter_visibility(self, *args) -> None:
        method = self.design_method_var.get()
        
        # Hide the entire filter parameters frame initially
        self.param_frame.pack_forget()
        
        # Only show filter parameters frame if needed
        if method in ['cheby1', 'cheby2', 'ellip']:
            self.param_frame.pack(fill=tk.X, pady=10)
            
            # Hide both parameter frames initially
            self.rp_frame.pack_forget()
            self.rs_frame.pack_forget()
            
            # Show relevant parameters
            if method in ['cheby1', 'ellip']:
                self.rp_frame.pack(fill=tk.X)
            if method in ['cheby2', 'ellip']:
                self.rs_frame.pack(fill=tk.X)

    def update_cutoff_fields(self) -> None:
        # Hide both frames
        self.single_cutoff_frame.pack_forget()
        self.double_cutoff_frame.pack_forget()
        
        # Show appropriate frame based on filter type
        if self.filter_type_var.get() in ['lowpass', 'highpass']:
            self.single_cutoff_frame.pack(fill=tk.X)
        else:
            self.double_cutoff_frame.pack(fill=tk.X)

    def update_code_display(self) -> None:
        try:
            # Get current parameters
            order = int(self.order_entry.get())
            fs = float(self.fs_entry.get())
            method = self.design_method_var.get()
            ftype = self.filter_type_var.get()
            
            code_lines = [
                "from scipy import signal",
                "import numpy as np",
                "from numpy.typing import NDArray",
                "",
                "def design_filter() -> tuple[NDArray, NDArray]:",
                f"    fs = {fs}  # Sampling frequency (Hz)",
                f"    order = {order}  # Filter order"
            ]
            
            # Add cutoff frequencies
            if ftype in ['bandpass', 'bandstop']:
                low = float(self.low_cutoff_entry.get())
                high = float(self.high_cutoff_entry.get())
                code_lines.append(f"    low_freq = {low}  # Low cutoff frequency (Hz)")
                code_lines.append(f"    high_freq = {high}  # High cutoff frequency (Hz)")
                code_lines.append(f"    wn = [2 * low_freq / fs, 2 * high_freq / fs]")
            else:
                cutoff = float(self.cutoff_entry.get())
                code_lines.append(f"    cutoff = {cutoff}  # Cutoff frequency (Hz)")
                code_lines.append(f"    wn = 2 * cutoff / fs")
            
            # Add filter design call based on method
            if method == 'butter':
                code_lines.append(f"    b, a = signal.butter(order, wn, btype='{ftype}')")
            elif method == 'cheby1':
                rp = float(self.rp_var.get())
                code_lines.append(f"    rp = {rp}  # Passband ripple (dB)")
                code_lines.append(f"    b, a = signal.cheby1(order, rp, wn, btype='{ftype}')")
            elif method == 'cheby2':
                rs = float(self.rs_var.get())
                code_lines.append(f"    rs = {rs}  # Stopband attenuation (dB)")
                code_lines.append(f"    b, a = signal.cheby2(order, rs, wn, btype='{ftype}')")
            elif method == 'ellip':
                rp = float(self.rp_var.get())
                rs = float(self.rs_var.get())
                code_lines.append(f"    rp = {rp}  # Passband ripple (dB)")
                code_lines.append(f"    rs = {rs}  # Stopband attenuation (dB)")
                code_lines.append(f"    b, a = signal.ellip(order, rp, rs, wn, btype='{ftype}')")
            elif method == 'bessel':
                code_lines.append(f"    b, a = signal.bessel(order, wn, btype='{ftype}', norm='mag')")
            
            code_lines.append("    return b, a")

            # Update text widget
            self.code_text.delete(1.0, tk.END)
            self.code_text.insert(1.0, '\n'.join(code_lines))
            
        except ValueError:
            self.code_text.delete(1.0, tk.END)
            self.code_text.insert(1.0, "# Invalid parameters")
    
    def _init_plots(self) -> None:
        self.fig = Figure(figsize=(10, 10))
        
        # Magnitude Response
        self.mag_ax = self.fig.add_subplot(221)
        self.mag_ax.set_title("Magnitude Response")
        self.mag_ax.set_xlabel("Frequency (Hz)")
        self.mag_ax.set_ylabel("Magnitude (dB)")
        self.mag_ax.grid(True)
        
        # Pole-Zero Plot
        self.pz_ax = self.fig.add_subplot(222)
        self.pz_ax.set_title("Pole-Zero Plot")
        self.pz_ax.set_xlabel("Real")
        self.pz_ax.set_ylabel("Imaginary")
        self.pz_ax.grid(True)
        
        # Phase Response
        self.phase_ax = self.fig.add_subplot(223)
        self.phase_ax.set_title("Phase Response")
        self.phase_ax.set_xlabel("Frequency (Hz)")
        self.phase_ax.set_ylabel("Phase (degrees)")
        self.phase_ax.grid(True)
        
        # Impulse Response
        self.imp_ax = self.fig.add_subplot(224)
        self.imp_ax.set_title("Impulse Response")
        self.imp_ax.set_xlabel("Time (s)")
        self.imp_ax.set_ylabel("Amplitude")
        self.imp_ax.grid(True)
        
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.plot_frame)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
    def design_filter(self) -> None:
        try:
            # Get parameters
            order = int(self.order_entry.get())
            fs = float(self.fs_entry.get())
            ftype = self.filter_type_var.get()
            method = self.design_method_var.get()
            rp = float(self.rp_var.get())
            rs = float(self.rs_var.get())
            
            # Parse cutoff frequencies
            if ftype in ['bandpass', 'bandstop']:
                low = float(self.low_cutoff_entry.get())
                high = float(self.high_cutoff_entry.get())
                wn = [low, high]
                cutoff_freqs = [low, high]
            else:
                cutoff = float(self.cutoff_entry.get())
                wn = cutoff
                cutoff_freqs = [cutoff]
            
            # Design filter based on method
            if method == 'butter':
                b, a = signal.butter(order, wn, btype=ftype, fs=fs)
            elif method == 'cheby1':
                b, a = signal.cheby1(order, rp, wn, btype=ftype, fs=fs)
            elif method == 'cheby2':
                b, a = signal.cheby2(order, rs, wn, btype=ftype, fs=fs)
            elif method == 'ellip':
                b, a = signal.ellip(order, rp, rs, wn, btype=ftype, fs=fs)
            elif method == 'bessel':
                b, a = signal.bessel(order, wn, btype=ftype, norm='mag', fs=fs)

            
            # Frequency response
            w, h = signal.freqz(b, a)
            freqs = w * fs / (2 * np.pi)
            mag = 20 * np.log10(np.abs(h))
            phase = np.unwrap(np.angle(h))
            phase = np.degrees(phase)
            
            # Pole-zero
            zeros, poles, _ = signal.tf2zpk(b, a)
            
            # Impulse response
            t = np.arange(0, 0.1, 1/fs)
            imp = np.zeros_like(t)
            imp[0] = 1
            imp_response = signal.lfilter(b, a, imp)
            
            # Update plots
            self.mag_ax.clear()
            self.phase_ax.clear()
            self.pz_ax.clear()
            self.imp_ax.clear()
            
            # Magnitude plot with cutoff lines
            self.mag_ax.semilogx(freqs, mag)
            ylim = self.mag_ax.get_ylim()
            for f in cutoff_freqs:
                self.mag_ax.axvline(f, color='r', linestyle='--', alpha=0.5)
            self.mag_ax.set_ylim(ylim)
            self.mag_ax.set_title("Magnitude Response")
            self.mag_ax.set_xlabel("Frequency (Hz)")
            self.mag_ax.set_ylabel("Magnitude (dB)")
            self.mag_ax.grid(True)
            
            # Phase plot with cutoff lines
            self.phase_ax.semilogx(freqs, phase)
            ylim = self.phase_ax.get_ylim()
            for f in cutoff_freqs:
                self.phase_ax.axvline(f, color='r', linestyle='--', alpha=0.5)
            self.phase_ax.set_ylim(ylim)
            self.phase_ax.set_title("Phase Response")
            self.phase_ax.set_xlabel("Frequency (Hz)")
            self.phase_ax.set_ylabel("Phase (degrees)")
            self.phase_ax.grid(True)
            
            # Pole-zero plot
            self.pz_ax.plot(np.real(zeros), np.imag(zeros), 'o', color='C0', label='Zeros')
            self.pz_ax.plot(np.real(poles), np.imag(poles), 'x', color='C0', label='Poles')
            self.pz_ax.set_title("Pole-Zero Plot")
            self.pz_ax.set_xlabel("Real")
            self.pz_ax.set_ylabel("Imaginary")
            self.pz_ax.legend()
            self.pz_ax.grid(True)
            
            # Draw unit circle
            circle = patches.Circle((0, 0), 1, fill=False, linestyle='--', color='gray')
            self.pz_ax.add_patch(circle)
            self.pz_ax.set_aspect('equal')
            
            # Set limits for pole-zero plot
            max_lim = max(
                abs(np.max(np.real(poles))) if len(poles) > 0 else 0,
                abs(np.max(np.real(zeros))) if len(zeros) > 0 else 0,
                abs(np.max(np.imag(poles))) if len(poles) > 0 else 0,
                abs(np.max(np.imag(zeros))) if len(zeros) > 0 else 0
            )
            lim = max(1.5, max_lim * 1.2)
            self.pz_ax.set_xlim(-lim, lim)
            self.pz_ax.set_ylim(-lim, lim)
            
            # Impulse response plot
            max_mag = np.max(np.abs(imp_response))
            index = len(imp_response) - 1
            for _ in range(len(imp_response)):
                if np.abs(imp_response[index]) < max_mag * 0.01:
                    index -= 1
                else:
                    break
                
            # self.imp_ax.plot(t[:index], imp_response[:index])
            markerline, stemlines, baseline = self.imp_ax.stem(t[:index], imp_response[:index])
            plt.setp(markerline, 'markerfacecolor', 'C0')
            plt.setp(baseline, 'color', 'grey', 'linewidth', 0.5)
            self.imp_ax.set_title("Impulse Response")
            self.imp_ax.set_xlabel("Time (sec)")
            self.imp_ax.set_ylabel("Amplitude")
            self.imp_ax.grid(True)
            
            self.fig.tight_layout()
            self.canvas.draw()

            # After successful filter design, update the code display
            self.update_code_display()
            
        except ValueError as e:
            messagebox.showerror("Error", "Invalid input parameters")
        except Exception as e:
            messagebox.showerror("Error", str(e))

def main() -> None:
    root = tk.Tk()
    root.geometry("1200x800")
    app = FilterDesignTool(root)
    root.mainloop()

if __name__ == "__main__":
    main()
