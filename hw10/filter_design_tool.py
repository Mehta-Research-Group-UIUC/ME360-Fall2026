"""Filter Design Lab: inspect digital IIR filters and apply them to local signals.

Run with Python 3.10+: python filter_design_tool.py [signal_file]
Requires PyQt5, NumPy, SciPy, and Matplotlib. MATLAB v7.3 files also require h5py.
CSV/TSV/TXT may have headers, a numeric time column (seconds or time_ms),
and signal columns. Array files may supply a time vector or scalar fs in Hz.
Enter the sampling rate in the app when the file does not provide one.
"""
from __future__ import annotations

import argparse
import csv
import importlib
import math
import sys
from array import array
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from itertools import chain
from pathlib import Path
from typing import Any, Callable, TypeVar

import numpy as np
from PyQt5 import QtCore, QtGui, QtWidgets
from matplotlib import rc_context
from matplotlib.axes import Axes
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg, NavigationToolbar2QT
from matplotlib.figure import Figure
from matplotlib.patches import Circle
from matplotlib.ticker import LogLocator
from numpy.typing import NDArray
from scipy import io, signal
from scipy.io import wavfile

FloatArray = NDArray[np.float64]
ComplexArray = NDArray[np.complex128]
IntArray = NDArray[np.intp]
T = TypeVar('T')

ECG_FILE = Path(__file__).resolve().with_name('ecg_signals.mat')
COURSE_RATE_HZ = 300.0  # The ME 360 Homework 11 ECG has no rate field.
SUPPORTED_EXTENSIONS = ('.mat', '.csv', '.tsv', '.txt', '.npy', '.npz', '.wav')
RATE_KEYS = {'fs', 'fshz', 'samplerate', 'sampleratehz', 'samplingrate',
             'samplingratehz', 'samplingfrequency', 'samplingfrequencyhz'}
TIME_KEYS = {'t', 'ts', 'time', 'times', 'timesec', 'timesecs', 'timeseconds',
             'seconds', 'sec', 'timestamp', 'timestamps', 'timestampss'}
TIME_MS_KEYS = {'tms', 'timems', 'milliseconds', 'timestampms', 'timestampsms'}
MAX_FILE_BYTES = 128 * 1024**2
MAX_PLOT_POINTS = 16_000
MAX_SPECTRUM_SAMPLES = 262_144

FILTER_TYPES = {'Low-pass': 'lowpass', 'High-pass': 'highpass',
                'Band-pass': 'bandpass', 'Band-stop': 'bandstop'}
FILTER_FAMILIES = {'Butterworth': 'butter', 'Chebyshev I': 'cheby1',
                   'Chebyshev II': 'cheby2', 'Elliptic': 'ellip', 'Bessel': 'bessel'}
EDGE_NAMES = {'butter': '−3 dB cutoff', 'cheby1': 'Passband edge',
              'cheby2': 'Stopband edge', 'ellip': 'Passband edge', 'bessel': '−3 dB cutoff'}
OFFLINE = 'Zero phase'
CAUSAL = 'Causal'
NO_REFERENCE = 'None'

BACKGROUND, WHITE, INK = '#edf2f6', '#ffffff', '#15263a'
MUTED, BORDER, GRID = '#53667b', '#dce5ed', '#e7edf3'
HEADER, HEADER_TEXT = '#14273c', '#bacbdc'
ACCENT, ACCENT_DARK, SELECTED = '#146b64', '#0e514c', '#e1f1ed'
INPUT_COLOR, OUTPUT_COLOR, REFERENCE_COLOR = '#747c87', '#ff7f0e', '#1f77b4'
STOPBAND_COLOR = '#6b7280'
MARKER_COLOR = '#7c3aed'
ERROR_COLOR, WAIT_COLOR = '#ab392a', '#82500a'


class SignalError(ValueError):
    """A file or setting needs attention before it can be used."""


@dataclass(frozen=True, slots=True)
class SignalChannel:
    name: str
    samples: FloatArray
    time_seconds: FloatArray | None = None
    sample_rate_hz: float | None = None
    rate_source: str = ''


@dataclass(frozen=True, slots=True)
class SignalSource:
    name: str
    channels: tuple[SignalChannel, ...]
    path: Path | None = None
    course_ecg: bool = False


@dataclass(frozen=True, slots=True)
class FilterSettings:
    sample_rate_hz: float
    order: int
    kind: str
    family: str
    edges_hz: tuple[float, ...]
    zero_phase: bool
    ripple_db: float | None = None
    attenuation_db: float | None = None


@dataclass(frozen=True, slots=True)
class FilterDesign:
    settings: FilterSettings
    sections: FloatArray
    zeros: ComplexArray
    poles: ComplexArray
    frequency_hz: FloatArray
    gain_db: FloatArray
    phase_degrees: FloatArray
    impulse: FloatArray


@dataclass(frozen=True, slots=True)
class FilterResult:
    design: FilterDesign
    source_name: str
    input_channel: SignalChannel
    reference: SignalChannel | None
    time_seconds: FloatArray
    output: FloatArray
    spectrum_frequency_hz: FloatArray
    input_spectrum: FloatArray
    output_spectrum: FloatArray
    spectrum_samples: int


def normalized_name(name: str) -> str:
    return ''.join(ch for ch in name.casefold() if ch.isalnum())


def positive_number(value: float, label: str) -> float:
    if not math.isfinite(value) or value <= 0:
        raise SignalError(f'{label} must be a finite number greater than zero.')
    return value


def time_vector_rate(time_seconds: FloatArray) -> float:
    if time_seconds.size < 2 or not np.all(np.isfinite(time_seconds)):
        raise SignalError('The time column needs at least two finite values.')
    intervals = np.diff(time_seconds)
    if not np.all(np.isfinite(intervals)):
        raise SignalError('Time values are too far apart to compute a finite sampling interval.')
    step = float(np.median(intervals))
    if step <= 0 or np.any(intervals <= 0):
        raise SignalError('Time values must increase strictly.')
    if not np.allclose(intervals, step, rtol=1e-3, atol=0.0):
        raise SignalError('Time values are not evenly spaced. Resample to a uniform grid before filtering.')
    return positive_number(1.0 / step, 'Rate inferred from timestamps')


def read_text_columns(path: Path) -> dict[str, np.ndarray]:
    def number(value: str) -> float:
        try:
            return float(value.strip())
        except ValueError:
            return math.nan

    def numeric_token(value: str) -> bool:
        if not value.strip():
            return True
        try:
            float(value.strip())
        except ValueError:
            return False
        return True

    with path.open(encoding='utf-8-sig', newline='') as stream:
        lines = (line for line in stream if line.strip() and not line.lstrip().startswith('#'))
        first_line = next(lines, None)
        if first_line is None:
            raise SignalError('The text file contains no data.')
        delimiter = next((value for value in ('\t', ',', ';') if value in first_line), None)
        all_lines = chain((first_line,), lines)
        rows = csv.reader(all_lines, delimiter=delimiter) if delimiter else (line.split() for line in all_lines)
        first_row = next(rows)
        width = len(first_row)
        has_header = not all(numeric_token(value) for value in first_row)
        labels = ([value.strip() or f'Column {i + 1}' for i, value in enumerate(first_row)]
                  if has_header else [f'Column {i + 1}' for i in range(width)])
        buffers = [array('d') for _ in labels]
        samples = rows if has_header else chain((first_row,), rows)
        for row in samples:
            if len(row) != width:
                raise SignalError('Every row must have the same number of columns. Check the delimiter or missing cells.')
            for column, value in zip(buffers, row):
                column.append(number(value))
    if not buffers or len(buffers[0]) < 2:
        raise SignalError('The file needs at least two sample rows.')
    columns: dict[str, np.ndarray] = {}
    for label, values in zip(labels, buffers):
        name = label
        suffix = 2
        while name in columns:
            name = f'{label} ({suffix})'
            suffix += 1
        values_array = np.frombuffer(values, dtype=np.float64)
        if np.any(np.isfinite(values_array)) or normalized_name(label) in RATE_KEYS | TIME_KEYS | TIME_MS_KEYS:
            columns[name] = values_array
    return columns


def read_mat_arrays(path: Path) -> dict[str, np.ndarray]:
    try:
        return {name: value for name, value in io.loadmat(path).items()
                if not name.startswith('__') and isinstance(value, np.ndarray)}
    except NotImplementedError:
        try:
            h5py = importlib.import_module('h5py')
        except ModuleNotFoundError as exc:
            raise SignalError('MATLAB v7.3 files require h5py. Install it or save the data as MATLAB v7.') from exc
        with h5py.File(path, 'r') as document:
            return {name: np.asarray(value) for name, value in document.items()
                    if isinstance(value, h5py.Dataset) and value.dtype.kind in 'iuf'}


def build_channels(arrays: dict[str, np.ndarray], *, sample_rows: bool = False,
                   fallback_rate_hz: float | None = None) -> tuple[SignalChannel, ...]:
    rate_hz: float | None = None
    time_seconds: FloatArray | None = None
    for name, array in arrays.items():
        key = normalized_name(name)
        if key in RATE_KEYS | TIME_KEYS | TIME_MS_KEYS and array.dtype.kind not in 'iuf':
            raise SignalError(f'“{name}” must contain real numeric values.')
        if key in RATE_KEYS:
            values = np.asarray(array, dtype=np.float64).reshape(-1)
            if values.size == 0 or not np.all(values == values[0]):
                raise SignalError(f'“{name}” must contain one constant sampling rate in Hz.')
            candidate_rate = positive_number(float(values[0]), name)
            if rate_hz is not None and not math.isclose(rate_hz, candidate_rate, rel_tol=1e-3):
                raise SignalError('The file contains conflicting sampling rates.')
            rate_hz = candidate_rate
        elif key in TIME_KEYS | TIME_MS_KEYS:
            values = np.asarray(array, dtype=np.float64).squeeze()
            if values.ndim != 1:
                raise SignalError(f'“{name}” must be a one-dimensional time vector.')
            candidate_time = values * .001 if key in TIME_MS_KEYS else values
            if time_seconds is not None and not np.array_equal(time_seconds, candidate_time):
                raise SignalError('The file contains multiple different time vectors.')
            time_seconds = candidate_time
    time_rate = time_vector_rate(time_seconds) if time_seconds is not None else None
    if time_rate is not None and rate_hz is not None and not math.isclose(time_rate, rate_hz, rel_tol=1e-3):
        raise SignalError('The sampling-rate field does not match the time vector.')
    channels: list[SignalChannel] = []
    taken: set[str] = {NO_REFERENCE}
    for name, array in arrays.items():
        if normalized_name(name) in RATE_KEYS | TIME_KEYS | TIME_MS_KEYS or array.dtype.kind not in 'iuf':
            continue
        data = np.asarray(array, dtype=np.float64).squeeze()
        if data.ndim == 1:
            data = data[:, None]
        elif data.ndim == 2 and not sample_rows:
            if time_seconds is not None and time_seconds.size in data.shape:
                if data.shape[0] != time_seconds.size:
                    data = data.T
            elif data.shape[0] < data.shape[1]:
                data = data.T
        if data.ndim != 2 or data.shape[0] < 2:
            continue
        aligned_time = time_seconds if time_seconds is not None and time_seconds.size == data.shape[0] else None
        if sample_rows and time_seconds is not None and aligned_time is None:
            raise SignalError('Signal columns must have the same number of samples as the time column.')
        channel_rate = time_rate if aligned_time is not None else rate_hz or fallback_rate_hz
        rate_source = ('Time vector' if aligned_time is not None else 'File metadata' if rate_hz is not None
                       else 'Course ECG preset' if fallback_rate_hz is not None else '')
        for i, samples in enumerate(data.T):
            label = name if data.shape[1] == 1 else f'{name} [{i + 1}]'
            channel_name = label
            suffix = 2
            while channel_name in taken:
                channel_name = f'{label} ({suffix})'
                suffix += 1
            taken.add(channel_name)
            channels.append(SignalChannel(channel_name, np.ascontiguousarray(samples),
                                          aligned_time, channel_rate, rate_source))
    if not channels:
        raise SignalError('No real numeric signal with at least two samples was found.')
    return tuple(channels)


def load_signal_file(path: Path) -> SignalSource:
    extension = path.suffix.casefold()
    if extension not in SUPPORTED_EXTENSIONS:
        raise SignalError('Choose a MAT, CSV, TSV, TXT, NPY, NPZ, or WAV file.')
    if path.stat().st_size > MAX_FILE_BYTES:
        raise SignalError('This file is larger than 128 MB. Load a shorter excerpt.')
    if extension == '.wav':
        rate_hz, audio = wavfile.read(path)
        if audio.dtype.kind == 'u':
            midpoint = (int(np.iinfo(audio.dtype).max) + 1) / 2
            samples = (audio.astype(np.float64) - midpoint) / midpoint
        elif audio.dtype.kind == 'i':
            samples = audio.astype(np.float64) / -int(np.iinfo(audio.dtype).min)
        else:
            samples = audio.astype(np.float64)
        channels = build_channels({'Audio': samples, 'fs': np.array(rate_hz)}, sample_rows=True)
        return SignalSource(path.name, channels, path)
    if extension == '.mat':
        arrays = read_mat_arrays(path)
    elif extension in {'.csv', '.tsv', '.txt'}:
        arrays = read_text_columns(path)
    elif extension == '.npy':
        arrays = {path.stem: np.load(path, allow_pickle=False)}
    else:
        with np.load(path, allow_pickle=False) as archive:
            arrays = {name: archive[name] for name in archive.files}
    course_ecg = (path.name.casefold() == ECG_FILE.name.casefold()
                  and {'ecg_clean', 'ecg_noisy'} <= arrays.keys()
                  and arrays['ecg_clean'].size == arrays['ecg_noisy'].size == 600)
    channels = build_channels(arrays, sample_rows=extension in {'.csv', '.tsv', '.txt'},
                              fallback_rate_hz=COURSE_RATE_HZ if course_ecg else None)
    return SignalSource(path.name, channels, path, course_ecg)


def example_signal() -> SignalSource:
    rate = COURSE_RATE_HZ
    t = np.arange(600) / rate
    clean = np.sin(2 * np.pi * 5 * t) + .25 * np.sin(2 * np.pi * 12 * t)
    noisy = clean + .40 * np.sin(2 * np.pi * 60 * t)
    return SignalSource('Synthetic example', (
        SignalChannel('Input', noisy, sample_rate_hz=rate, rate_source='Example preset'),
        SignalChannel('Clean reference', clean, sample_rate_hz=rate, rate_source='Example preset'),
    ))


def validate_settings(settings: FilterSettings) -> None:
    nyquist = positive_number(settings.sample_rate_hz, 'Sampling rate') / 2
    if settings.kind not in FILTER_TYPES.values() or settings.family not in FILTER_FAMILIES.values():
        raise SignalError('Choose a supported filter type and design family.')
    if not 1 <= settings.order <= 20:
        raise SignalError('Filter order must be an integer from 1 to 20.')
    expected_edges = 2 if settings.kind in {'bandpass', 'bandstop'} else 1
    if len(settings.edges_hz) != expected_edges or any(
        not math.isfinite(edge) or not 0 < edge < nyquist for edge in settings.edges_hz
    ):
        raise SignalError(f'Enter edge frequencies above 0 and below Nyquist ({nyquist:g} Hz).')
    if expected_edges == 2 and settings.edges_hz[0] >= settings.edges_hz[1]:
        raise SignalError('The low edge must be below the high edge.')
    if settings.family in {'cheby1', 'ellip'}:
        positive_number(settings.ripple_db if settings.ripple_db is not None else math.nan, 'Passband ripple')
    if settings.family in {'cheby2', 'ellip'}:
        positive_number(settings.attenuation_db if settings.attenuation_db is not None else math.nan,
                        'Stopband attenuation')
    if settings.family == 'ellip' and (settings.attenuation_db or 0) <= (settings.ripple_db or 0):
        raise SignalError('Stopband attenuation must exceed passband ripple for an elliptic filter.')


def design_arguments(settings: FilterSettings) -> tuple[str, dict[str, object]]:
    wn = list(settings.edges_hz) if len(settings.edges_hz) == 2 else settings.edges_hz[0]
    kwargs: dict[str, object] = {'N': settings.order, 'Wn': wn, 'btype': settings.kind,
                                 'fs': settings.sample_rate_hz, 'output': 'sos'}
    if settings.family in {'cheby1', 'ellip'}:
        kwargs['rp'] = settings.ripple_db
    if settings.family in {'cheby2', 'ellip'}:
        kwargs['rs'] = settings.attenuation_db
    if settings.family == 'bessel':
        kwargs['norm'] = 'mag'
    return settings.family, kwargs


def design_filter(settings: FilterSettings) -> FilterDesign:
    validate_settings(settings)
    method, arguments = design_arguments(settings)
    constructor: Callable[..., Any] = getattr(signal, method)
    sections = np.asarray(constructor(**arguments), dtype=np.float64)
    if not np.all(np.isfinite(sections)):
        raise SignalError('These settings did not produce finite coefficients. Try a lower order or wider band.')
    zeros, poles, _ = signal.sos2zpk(sections)
    # Odd-order designs contain a padded pole/zero pair at the origin in SOS form.
    padded = min(np.count_nonzero(zeros == 0), np.count_nonzero(poles == 0))
    if padded:
        zeros = np.delete(zeros, np.flatnonzero(zeros == 0)[:padded])
        poles = np.delete(poles, np.flatnonzero(poles == 0)[:padded])
    max_radius = float(np.max(np.abs(poles)))
    if not max_radius < 1.0:
        raise SignalError('This filter is numerically unstable. Try a lower order or wider band.')
    nyquist = settings.sample_rate_hz / 2
    first_frequency = max(float(np.finfo(np.float64).tiny), min(nyquist / 1_000, settings.edges_hz[0] / 10))
    frequencies = [np.geomspace(first_frequency, nyquist, 4_096), np.asarray(settings.edges_hz)]
    if len(settings.edges_hz) == 2:
        low, high = settings.edges_hz
        width = high - low
        frequencies.append(np.linspace(max(first_frequency, low - width), min(nyquist, high + width), 1_024))
    nulls = np.abs(np.angle(zeros)) * settings.sample_rate_hz / (2 * np.pi)
    frequencies.append(nulls[(nulls >= first_frequency) & (nulls <= nyquist)])
    frequency_hz = np.unique(np.concatenate(frequencies))
    _, response = signal.sosfreqz(sections, worN=frequency_hz, fs=settings.sample_rate_hz)
    magnitude = np.abs(response)
    if not np.all(np.isfinite(magnitude)):
        raise SignalError('The frequency response is not finite for these settings.')
    gain = 20 * np.log10(np.maximum(magnitude, np.finfo(np.float64).tiny))
    valid_phase = magnitude > 1e-12
    phase = np.full_like(magnitude, np.nan)
    phase[valid_phase] = np.degrees(np.unwrap(np.angle(response[valid_phase])))
    count = min(2_048, max(64, math.ceil(math.log(1e-4) / math.log(max_radius)))) if max_radius > 0 else 64
    pulse = np.zeros(count, dtype=np.float64)
    pulse[0] = 1
    impulse = signal.sosfilt(sections, pulse)
    return FilterDesign(settings, sections, zeros, poles, frequency_hz, gain, phase, impulse)


def filter_samples(samples: FloatArray, design: FilterDesign) -> FloatArray:
    if not np.all(np.isfinite(samples)):
        raise SignalError('The selected signal has missing or non-finite samples. Fix them in the source file.')
    sos = design.sections
    if design.settings.zero_phase:
        # SciPy's default SOS forward/backward padding includes origin corrections.
        pad_length = 3 * (2 * len(sos) + 1 - min(np.count_nonzero(sos[:, 2] == 0),
                                                  np.count_nonzero(sos[:, 5] == 0)))
        if samples.size <= pad_length:
            raise SignalError(f'Zero-phase filtering needs at least {pad_length + 1} samples for this design. '
                              'Choose Causal processing or a lower order.')
        result = signal.sosfiltfilt(sos, samples)
    else:
        result = signal.sosfilt(sos, samples)
    if not np.all(np.isfinite(result)):
        raise SignalError('Filtering produced non-finite values. Try a lower order or rescale the input.')
    return np.asarray(result, dtype=np.float64)


def amplitude_spectrum(samples: FloatArray, rate_hz: float) -> tuple[FloatArray, FloatArray]:
    count = samples.size
    window = signal.windows.hann(count, sym=False) if count >= 4 else np.ones(count)
    magnitude = np.abs(np.fft.rfft(samples * window)) / window.sum()
    magnitude[1:-1 if count % 2 == 0 else None] *= 2
    return np.fft.rfftfreq(count, d=1 / rate_hz), magnitude


def compatible_reference(input_channel: SignalChannel, reference: SignalChannel) -> bool:
    if reference.name == input_channel.name or reference.samples.shape != input_channel.samples.shape:
        return False
    if input_channel.sample_rate_hz is not None and reference.sample_rate_hz is not None:
        if not math.isclose(input_channel.sample_rate_hz, reference.sample_rate_hz, rel_tol=1e-3):
            return False
    if input_channel.time_seconds is not None and reference.time_seconds is not None:
        if input_channel.time_seconds is reference.time_seconds:
            return True
        return bool(np.allclose(input_channel.time_seconds, reference.time_seconds, rtol=0, atol=1e-8))
    return True


def analyze_signal(settings: FilterSettings, source_name: str, input_channel: SignalChannel,
                   reference: SignalChannel | None) -> FilterResult:
    design = design_filter(settings)
    output = filter_samples(input_channel.samples, design)
    if reference is not None and not np.all(np.isfinite(reference.samples)):
        raise SignalError('The reference signal has missing or non-finite values. Choose another reference or None.')
    count = input_channel.samples.size
    time_seconds = input_channel.time_seconds
    if time_seconds is None:
        time_seconds = np.arange(count, dtype=np.float64) / settings.sample_rate_hz
    spectrum_count = min(count, MAX_SPECTRUM_SAMPLES)
    start = (count - spectrum_count) // 2
    segment = slice(start, start + spectrum_count)
    frequencies, input_magnitude = amplitude_spectrum(input_channel.samples[segment], settings.sample_rate_hz)
    _, output_magnitude = amplitude_spectrum(output[segment], settings.sample_rate_hz)
    return FilterResult(design, source_name, input_channel, reference, time_seconds, output,
                        frequencies, input_magnitude, output_magnitude, spectrum_count)


def plot_indices(samples: FloatArray) -> IntArray | slice:
    """Keep both extrema in each bucket to preserve peaks in large previews."""
    count = samples.size
    if count <= MAX_PLOT_POINTS:
        return slice(None)
    bucket_size = math.ceil(count / (MAX_PLOT_POINTS // 2 - 1))
    bucket_count = count // bucket_size
    buckets = samples[:bucket_count * bucket_size].reshape(bucket_count, bucket_size)
    offsets = np.arange(bucket_count) * bucket_size
    minima = offsets + buckets.argmin(axis=1)
    maxima = offsets + buckets.argmax(axis=1)
    tail_start = bucket_count * bucket_size
    tail = samples[tail_start:]
    indices = [0, count - 1]
    if tail.size:
        indices += [tail_start + int(tail.argmin()), tail_start + int(tail.argmax())]
    return np.unique(np.concatenate((minima, maxima, indices)))


def python_example(settings: FilterSettings) -> str:
    method, arguments = design_arguments(settings)
    arguments_text = ',\n'.join(f'        {name}={value!r}' for name, value in arguments.items())
    apply = 'sosfiltfilt' if settings.zero_phase else 'sosfilt'
    return ('import numpy as np\nfrom numpy.typing import NDArray\nfrom scipy import signal\n\n\n'
            'def design_filter() -> NDArray[np.float64]:\n'
            f'    return signal.{method}(\n{arguments_text},\n    )\n\n\n'
            'def filter_signal(samples: NDArray[np.float64]) -> NDArray[np.float64]:\n'
            f'    return signal.{apply}(design_filter(), samples)\n')


def export_signal_csv(path: Path, result: FilterResult) -> None:
    columns = [result.time_seconds, result.input_channel.samples, result.output]
    header = 'time_s,input,filtered'
    if result.reference is not None:
        columns.append(result.reference.samples)
        header += ',reference'
    np.savetxt(path, np.column_stack(columns), delimiter=',', header=header, comments='', fmt='%.17g')


class PlotTab(QtWidgets.QWidget):
    """A cached chart surface with independent pan and zoom controls."""

    def __init__(self, title: str, layout: str) -> None:
        super().__init__()
        panel = QtWidgets.QVBoxLayout(self)
        panel.setContentsMargins(17, 15, 17, 10)
        panel.setSpacing(6)
        heading = QtWidgets.QHBoxLayout()
        title_label = QtWidgets.QLabel(title)
        title_label.setObjectName('plotTitle')
        heading.addWidget(title_label, 1)
        self.zoom_button = QtWidgets.QPushButton('Zoom')
        self.pan_button = QtWidgets.QPushButton('Pan')
        self.reset_button = QtWidgets.QPushButton('Reset view')
        for button in (self.zoom_button, self.pan_button, self.reset_button):
            button.setProperty('role', 'tool')
            heading.addWidget(button)
        self.zoom_button.setCheckable(True)
        self.pan_button.setCheckable(True)
        panel.addLayout(heading)
        self.subtitle = QtWidgets.QLabel('Open a signal file to begin')
        self.subtitle.setObjectName('secondary')
        self.subtitle.setTextFormat(QtCore.Qt.PlainText)
        self.subtitle.setWordWrap(True)
        panel.addWidget(self.subtitle)

        # Diagnostics needs room for two sets of tick labels side by side.
        with rc_context({'figure.autolayout': False, 'figure.constrained_layout.w_pad': .16,
                         'figure.constrained_layout.h_pad': .16, 'figure.constrained_layout.wspace': .12}):
            self.figure = Figure(figsize=(8, 5.8), dpi=100, facecolor=WHITE,
                                 constrained_layout=layout == 'diagnostics')
        if layout == 'diagnostics':
            grid = self.figure.add_gridspec(1, 2, width_ratios=(1, 1.25))
            first, second = self.figure.add_subplot(grid[0, 0]), self.figure.add_subplot(grid[0, 1])
        else:
            grid = self.figure.add_gridspec(2, 1, height_ratios=(2.1, 1))
            first = self.figure.add_subplot(grid[0, 0])
            second = self.figure.add_subplot(grid[1, 0], sharex=first if layout == 'response' else None)
            self.figure.subplots_adjust(left=.105, right=.98, bottom=.10, top=.92,
                                        hspace=.44 if layout == 'response' else .57)
        self.axes: tuple[Axes, Axes] = (first, second)
        self.canvas = FigureCanvasQTAgg(self.figure)
        self.canvas.setMinimumSize(320, 280)
        self.canvas.setFocusPolicy(QtCore.Qt.ClickFocus)
        panel.addWidget(self.canvas, 1)
        self.navigation = NavigationToolbar2QT(self.canvas, self, coordinates=False)
        self.navigation.hide()
        self.zoom_button.clicked.connect(self._zoom)
        self.pan_button.clicked.connect(self._pan)
        self.reset_button.clicked.connect(self.navigation.home)
        for ax in self.axes:
            ax.set_axis_off()

    def show_empty(self, message: str, subtitle: str) -> None:
        self.subtitle.setText(subtitle)
        self.axes[0].set_xscale('linear')
        for ax in self.axes:
            ax.clear()
            ax.set_axis_off()
        self.axes[0].text(.5, .40, message, color=MUTED, fontsize=13, ha='center',
                          va='center', transform=self.axes[0].transAxes)
        self.redraw()

    def _zoom(self) -> None:
        self.navigation.zoom()
        self._sync_navigation()

    def _pan(self) -> None:
        self.navigation.pan()
        self._sync_navigation()

    def _sync_navigation(self) -> None:
        mode = str(self.navigation.mode)
        self.zoom_button.setChecked(mode == 'zoom rect')
        self.pan_button.setChecked(mode == 'pan/zoom')

    def redraw(self) -> None:
        self.navigation.update()
        self.navigation.push_current()
        self.canvas.draw_idle()


def style_axis(ax: Axes, title: str, ylabel: str, xlabel: str = '') -> None:
    ax.set_axis_on()
    ax.set_facecolor(WHITE)
    ax.set_title(title, loc='left', fontsize=11, fontweight='semibold', color=INK, pad=8)
    ax.set_xlabel(xlabel, color=MUTED, fontsize=9, labelpad=5)
    ax.set_ylabel(ylabel, color=MUTED, fontsize=9, labelpad=5)
    ax.tick_params(colors=MUTED, labelsize=8.5, length=3)
    ax.grid(color=GRID, linewidth=.75)
    ax.set_axisbelow(True)
    for side in ('left', 'bottom'):
        ax.spines[side].set_color(BORDER)
    for side in ('top', 'right'):
        ax.spines[side].set_visible(False)


def plot_trace(ax: Axes, x: FloatArray, y: FloatArray, **style: Any) -> None:
    points = plot_indices(y)
    ax.plot(x[points], y[points], **style)


def shade_stopbands(ax: Axes, settings: FilterSettings) -> None:
    """Dim the nominal attenuation regions within the displayed frequency range."""
    left, right = ax.get_xlim()
    low = settings.edges_hz[0]
    spans: tuple[tuple[float, float], ...]
    if settings.kind == 'lowpass':
        spans = ((low, right),)
    elif settings.kind == 'highpass':
        spans = ((left, low),)
    elif settings.kind == 'bandpass':
        spans = ((left, low), (settings.edges_hz[1], right))
    else:  # Band-stop
        spans = ((low, settings.edges_hz[1]),)

    for start, end in spans:
        start, end = max(left, start), min(right, end)
        if start < end:
            ax.axvspan(start, end, facecolor=STOPBAND_COLOR, alpha=.14, linewidth=0)


def draw_frequency_markers(ax: Axes, frequencies: tuple[float, ...], *, labels: bool = True) -> None:
    """Draw user-chosen reference frequencies that fall within the displayed range."""
    left, right = ax.get_xlim()
    for frequency in frequencies:
        if left < frequency < right:
            ax.axvline(frequency, color=MARKER_COLOR, linestyle=':', linewidth=1.2, zorder=3)
            if labels:
                ax.text(frequency, .98, f' {frequency:g} Hz', transform=ax.get_xaxis_transform(),
                        color=MARKER_COLOR, fontsize=7.5, rotation=90, ha='right', va='top', zorder=6)


def draw_signal_comparison(tab: PlotTab, result: FilterResult, markers: tuple[float, ...] = ()) -> None:
    wave, spectrum = tab.axes
    wave.clear()
    spectrum.clear()
    time = result.time_seconds
    count = time.size
    rate = result.design.settings.sample_rate_hz
    tab.subtitle.setText(f'{result.input_channel.name}  ·  {count:,} samples  ·  {rate:g} Hz'
                         '  ·  Shading marks the nominal stopband')
    style_axis(wave, 'Time domain', 'Amplitude', 'Time (s)')
    plot_trace(wave, time, result.input_channel.samples, color=INPUT_COLOR, label='Input', linewidth=1.05, zorder=2)
    plot_trace(wave, time, result.output, color=OUTPUT_COLOR, label='Filtered', linewidth=1.55, zorder=4)
    if result.reference is not None:
        plot_trace(wave, time, result.reference.samples, color=REFERENCE_COLOR, label='Reference',
                   linewidth=.95, linestyle=(0, (4, 3)), zorder=5)
    wave.set_xlim(float(time[0]), float(time[-1]))
    wave.legend(loc='upper right', ncol=3, fontsize=8.5, framealpha=.95, edgecolor=BORDER,
                labelcolor=INK, handlelength=1.8, columnspacing=1.1)
    title = 'One-sided amplitude spectrum'
    if result.spectrum_samples != count:
        title += f'  ·  center {result.spectrum_samples:,} samples'
    style_axis(spectrum, title, 'Amplitude', 'Frequency (Hz)')
    # DC has no position on a logarithmic axis. Keep it in the FFT data.
    frequency = result.spectrum_frequency_hz[1:]
    spectrum.set_xscale('log')
    spectrum.xaxis.set_minor_locator(LogLocator(base=10, subs=(2, 5)))
    marker = 'o' if frequency.size == 1 else None
    plot_trace(spectrum, frequency, result.input_spectrum[1:], color=INPUT_COLOR, linewidth=1.05, marker=marker)
    plot_trace(spectrum, frequency, result.output_spectrum[1:], color=OUTPUT_COLOR, linewidth=1.45, marker=marker)
    first = float(frequency[0])
    spectrum.set_xlim(first / 2 if frequency.size == 1 else first, rate / 2)
    spectrum.set_ylim(bottom=0)
    shade_stopbands(spectrum, result.design.settings)
    draw_frequency_markers(spectrum, markers)
    tab.redraw()


def draw_input_preview(tab: PlotTab, channel: SignalChannel) -> None:
    wave, spectrum = tab.axes
    wave.clear()
    spectrum.clear()
    tab.subtitle.setText(f'{channel.name}  ·  {channel.samples.size:,} samples  ·  sampling rate needed')
    style_axis(wave, 'Input preview', 'Amplitude', 'Sample index')
    samples = channel.samples
    if not np.all(np.isfinite(samples)):
        samples = np.where(np.isfinite(samples), samples, np.nan)
    points = plot_indices(np.nan_to_num(samples))
    indices = np.arange(samples.size)[points]
    wave.plot(indices, samples[points], color=INPUT_COLOR, linewidth=1.05)
    wave.set_xlim(0, samples.size - 1)
    spectrum.set_axis_off()
    spectrum.text(.5, .5, 'Enter the sampling rate to filter and view the spectrum.', color=MUTED,
                  ha='center', va='center', fontsize=10, transform=spectrum.transAxes)
    tab.redraw()


def draw_filter_response(tab: PlotTab, design: FilterDesign, markers: tuple[float, ...] = ()) -> None:
    gain, phase = tab.axes
    gain.set_xscale('linear')
    gain.clear()
    phase.clear()
    freq = design.frequency_hz
    gain_db = design.gain_db
    phase_degrees = design.phase_degrees
    mode = 'Causal (one forward pass)'
    if design.settings.zero_phase:
        # The backward pass contributes the conjugate response, so magnitudes
        # multiply and phases cancel wherever the phase is defined.
        gain_db = 2 * gain_db
        phase_degrees = np.where(np.isfinite(phase_degrees), 0.0, np.nan)
        mode = 'Zero phase (forward and backward)'
        phase.set_ylim(-1, 1)
        phase.set_yticks([0])
    tab.subtitle.setText(f'{mode}  ·  Shading marks the nominal stopband')
    style_axis(gain, 'Magnitude response', 'Gain (dB)')
    style_axis(phase, 'Unwrapped phase', 'Phase (°)', 'Frequency (Hz)')
    gain.set_xscale('log')
    gain.xaxis.set_minor_locator(LogLocator(base=10, subs=(2, 5)))
    gain.set_xlim(float(freq[0]), float(freq[-1]))
    gain.set_ylim(-100, max(3.0, float(np.max(gain_db)) + 2))
    gain.tick_params(labelbottom=False)
    gain.plot(freq, gain_db, color=OUTPUT_COLOR, linewidth=1.65)
    phase.plot(freq, phase_degrees, color=OUTPUT_COLOR, linewidth=1.65)
    for ax in (gain, phase):
        shade_stopbands(ax, design.settings)
    draw_frequency_markers(gain, markers)
    draw_frequency_markers(phase, markers, labels=False)
    tab.redraw()


def draw_filter_diagnostics(tab: PlotTab, design: FilterDesign) -> None:
    zplane, impulse = tab.axes
    zplane.clear()
    impulse.clear()
    radius = float(np.max(np.abs(design.poles)))
    tab.subtitle.setText(f'One forward pass  ·  order {len(design.poles)}  ·  maximum pole radius {radius:.6f}')
    style_axis(zplane, 'Poles and zeros', 'Imaginary', 'Real')
    zplane.add_patch(Circle((0, 0), 1, fill=False, edgecolor='#9aacbf', linestyle='--', linewidth=1.1))
    zplane.scatter(design.zeros.real, design.zeros.imag, facecolors='none', edgecolors=OUTPUT_COLOR,
                   s=38, linewidths=1.35, label='Zeros', zorder=3)
    zplane.scatter(design.poles.real, design.poles.imag, color=OUTPUT_COLOR, marker='x',
                   s=43, linewidths=1.6, label='Poles', zorder=4)
    limit = max(1.17, 1.1 * float(np.max(np.abs(design.zeros)))) if design.zeros.size else 1.17
    zplane.set(xlim=(-limit, limit), ylim=(-limit, limit), aspect='equal', anchor='N')
    zplane.legend(loc='upper center', bbox_to_anchor=(.5, -.24), borderaxespad=0,
                  ncol=2, fontsize=8.5, edgecolor=BORDER, labelcolor=INK)
    style_axis(impulse, 'Impulse response', 'Amplitude', 'Time (s)')
    time = np.arange(design.impulse.size) / design.settings.sample_rate_hz
    impulse.vlines(time, 0, design.impulse, color=OUTPUT_COLOR, linewidth=.9)
    impulse.axhline(0, color=BORDER, linewidth=.8)
    if time.size <= 180:
        impulse.scatter(time, design.impulse, color=OUTPUT_COLOR, s=5)
    impulse.margins(x=.03)
    tab.redraw()



@dataclass(frozen=True, slots=True)
class FilterRequest:
    revision: int
    settings: FilterSettings
    source_name: str
    channel: SignalChannel
    reference: SignalChannel | None
    description: str

    def calculate(self) -> FilterResult:
        return analyze_signal(self.settings, self.source_name, self.channel, self.reference)


@dataclass(frozen=True, slots=True)
class ActiveJob:
    kind: str
    future: Future[Any]
    accept: Callable[[Any], None]
    error_prefix: str


class JobEvents(QtCore.QObject):
    # A queued Qt signal delivers a completed Future to the GUI thread.
    finished = QtCore.pyqtSignal(object)


class SettingComboBox(QtWidgets.QComboBox):
    """Keep Qt's popup behavior and draw a lightweight chevron."""

    def __init__(self, options: tuple[str, ...] = ()) -> None:
        super().__init__()
        self.addItems(options)
        self.setSizeAdjustPolicy(QtWidgets.QComboBox.AdjustToMinimumContentsLengthWithIcon)
        self.setMaxVisibleItems(8)
        view = QtWidgets.QListView()
        view.setSpacing(2)
        view.setMouseTracking(True)
        self.setView(view)
        view.window().setObjectName('settingPopup')

    def paintEvent(self, event: QtGui.QPaintEvent | None) -> None:
        super().paintEvent(event)
        painter = QtGui.QPainter(self)
        painter.setRenderHint(QtGui.QPainter.Antialiasing)
        color = ACCENT if self.isEnabled() and self.hasFocus() else MUTED
        pen = QtGui.QPen(QtGui.QColor(color), 1.6)
        pen.setCapStyle(QtCore.Qt.RoundCap)
        pen.setJoinStyle(QtCore.Qt.RoundJoin)
        painter.setPen(pen)
        x, y = self.width() - 14, self.height() / 2
        dy = -2.0 if self.view().isVisible() else 2.0
        painter.drawPolyline(QtGui.QPolygonF([
            QtCore.QPointF(x - 4, y - dy), QtCore.QPointF(x, y + dy), QtCore.QPointF(x + 4, y - dy),
        ]))
        painter.end()


def stylesheet() -> str:
    return f'''
    QWidget {{ color:{INK}; }}
    QMainWindow, QWidget#windowBody {{ background:{BACKGROUND}; }}
    QFrame#header {{ background:{HEADER}; }}
    QLabel#headerTitle {{ color:{WHITE}; font-size:26px; font-weight:700; }}
    QLabel#headerSubtitle {{ color:{HEADER_TEXT}; font-size:13px; }}
    QFrame#sidebar, QWidget#fields, QScrollArea, QTabWidget::pane {{ background:{WHITE}; }}
    QFrame#sidebar {{ border:1px solid {BORDER}; border-radius:7px; }}
    QScrollArea {{ border:0; }}
    QLabel#section {{ color:{ACCENT_DARK}; font-size:12px; font-weight:700; }}
    QLabel#fileName {{ font-size:15px; font-weight:700; }}
    QLabel#fieldLabel {{ font-size:12px; font-weight:600; }}
    QLabel#plotTitle {{ font-size:19px; font-weight:700; }}
    QLabel#secondary, QLabel#footer {{ color:{MUTED}; font-size:12px; }}
    QPushButton {{ border:1px solid {BORDER}; border-radius:7px; padding:6px 12px;
                   background:#f4f8fa; font-size:13px; }}
    QPushButton:hover {{ background:#e4eef3; border-color:#b9ccd8; }}
    QPushButton:pressed {{ background:#d3e4eb; }}
    QPushButton:focus {{ border-color:{ACCENT}; }}
    QPushButton:checked {{ background:{SELECTED}; color:{ACCENT_DARK}; border-color:#a0cfc5; }}
    QPushButton:disabled {{ color:#788998; background:#e8eef1; border-color:{BORDER}; }}
    QPushButton[role="primary"] {{ background:{ACCENT}; color:{WHITE}; font-weight:700; border-color:{ACCENT}; padding:9px 15px; }}
    QPushButton[role="primary"]:hover {{ background:{ACCENT_DARK}; }}
    QPushButton[role="primary"]:pressed {{ background:#0b443f; }}
    QPushButton[role="primary"]:focus {{ border-color:#8ac7bd; }}
    QPushButton[role="primary"]:disabled {{ background:#dce7e6; color:#647775; border-color:#dce7e6; }}
    QPushButton[role="header"] {{ background:#233d55; border-color:#36516a; color:{WHITE}; padding:9px 14px; }}
    QPushButton[role="header"]:hover {{ background:#31546e; }}
    QPushButton[role="header"]:disabled {{ color:#8ea5b7; }}
    QPushButton[role="tool"] {{ padding:5px 11px; }}
    QLineEdit,QComboBox {{ border:1px solid {BORDER}; border-radius:7px; padding:6px 9px;
                          background:#f7fafc; min-height:18px; font-size:13px; }}
    QComboBox {{ combobox-popup:0; padding-right:3px; }}
    QComboBox:hover {{ background:#f0f6f8; border-color:#a8bdcb; }}
    QComboBox:on {{ background:{WHITE}; border-color:{ACCENT}; }}
    QComboBox::drop-down {{ subcontrol-origin:border; subcontrol-position:top right; width:26px; border:0; }}
    QComboBox::down-arrow {{ image:none; width:0; height:0; }}
    QLineEdit:focus,QComboBox:focus {{ border-color:{ACCENT}; background:{WHITE}; }}
    QLineEdit[readOnly="true"],QComboBox:disabled {{ background:#edf2f6; color:{MUTED}; }}
    QFrame#settingPopup {{ background:{WHITE}; border:1px solid #b8ccd6; border-radius:8px; }}
    QComboBox QListView {{ background:{WHITE}; border:1px solid #b8ccd6; border-radius:8px;
                          padding:4px; outline:0; font-size:13px;
                          selection-background-color:{SELECTED}; selection-color:{ACCENT_DARK}; }}
    QComboBox QListView::item {{ border-radius:5px; padding:5px 9px; min-height:17px; }}
    QComboBox QListView::item:hover {{ background:#eef5f6; }}
    QComboBox QListView::item:selected {{ background:{SELECTED}; color:{ACCENT_DARK}; }}
    QTabWidget::pane {{ border:1px solid {BORDER}; border-radius:8px; }}
    QTabBar::tab {{ background:transparent; color:{MUTED}; border:1px solid transparent; border-radius:8px;
                   min-width:80px; padding:7px 12px; margin:0 4px 5px 0; font-size:13px; font-weight:600; }}
    QTabBar::tab:hover {{ background:#e2ebf1; color:{INK}; }}
    QTabBar::tab:selected {{ background:{SELECTED}; color:{ACCENT_DARK}; border-color:#b4d9cf; }}
    QTabBar::tab:focus {{ border-color:{ACCENT}; }}
    QScrollBar:vertical {{ width:8px; background:transparent; margin:2px 0; }}
    QScrollBar::handle:vertical {{ background:#c1d1db; min-height:26px; border-radius:4px; }}
    QScrollBar::handle:vertical:hover {{ background:#9eb3c2; }}
    QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{ height:0; }}
    QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical {{ background:transparent; }}
    QPlainTextEdit {{ background:{HEADER}; color:#e8f2fa; border:0; border-radius:5px; padding:12px;
                      selection-background-color:{ACCENT}; }}
    QFrame#divider {{ color:{BORDER}; }}
    '''


class FilterDesignTool(QtWidgets.QMainWindow):
    """Run file and filter jobs without blocking Qt's controls."""

    def __init__(self, initial_file: Path | None = None) -> None:
        super().__init__()
        self.source: SignalSource | None = None
        self.channels: dict[str, SignalChannel] = {}
        self.result: FilterResult | None = None
        self.revision = 0
        self.result_revision = -1
        self.rendered_tabs: set[int] = set()
        self._updating_form = False
        self._closing = False
        self._job: ActiveJob | None = None
        self._filter_in_flight: FilterRequest | None = None
        self._pending_filter: FilterRequest | None = None
        self._worker = ThreadPoolExecutor(max_workers=1, thread_name_prefix='filter-lab')
        self._events = JobEvents()
        self._events.finished.connect(self._job_finished, QtCore.Qt.QueuedConnection)
        self._update_timer = QtCore.QTimer(self)
        self._update_timer.setSingleShot(True)
        self._update_timer.setInterval(30)
        self._update_timer.timeout.connect(self._update_filter)
        self._plot_timer = QtCore.QTimer(self)
        self._plot_timer.setSingleShot(True)
        self._plot_timer.setInterval(22)
        self._plot_timer.timeout.connect(self._draw_current_tab)

        self.setWindowTitle('Filter Design Lab')
        self.setMinimumSize(970, 650)
        screen = QtWidgets.QApplication.primaryScreen()
        area = screen.availableGeometry() if screen is not None else QtCore.QRect(0, 0, 1440, 900)
        self.resize(min(1360, max(970, area.width() - 80)), min(850, max(650, area.height() - 60)))
        self.move(area.x() + 35, area.y() + 20)
        self.setStyleSheet(stylesheet())
        self._build_interface()
        self._connect_controls()
        application = QtWidgets.QApplication.instance()
        if application is not None:
            application.installEventFilter(self)
        self._update_filter_fields()
        self._update_rate_info()
        self._sync_actions()
        self._schedule_plot()
        if initial_file is None:
            QtCore.QTimer.singleShot(35, self._open_example)
        else:
            QtCore.QTimer.singleShot(35, lambda: self._open_path(initial_file))

    @property
    def busy(self) -> bool:
        return self._job is not None

    def _build_interface(self) -> None:
        window = QtWidgets.QWidget()
        window.setObjectName('windowBody')
        self.setCentralWidget(window)
        outer = QtWidgets.QVBoxLayout(window)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)
        header = QtWidgets.QFrame()
        header.setObjectName('header')
        header_layout = QtWidgets.QHBoxLayout(header)
        header_layout.setContentsMargins(23, 13, 23, 13)
        header_text = QtWidgets.QVBoxLayout()
        header_text.setSpacing(3)
        title = QtWidgets.QLabel('Filter Design Lab')
        title.setObjectName('headerTitle')
        subtitle = QtWidgets.QLabel('Design digital filters and see what they change.')
        subtitle.setObjectName('headerSubtitle')
        header_text.addWidget(title)
        header_text.addWidget(subtitle)
        header_layout.addLayout(header_text, 1)
        self.example_button = QtWidgets.QPushButton('Load example')
        self.example_button.setProperty('role', 'header')
        self.open_button = QtWidgets.QPushButton('Open signal…')
        self.open_button.setProperty('role', 'primary')
        header_layout.addWidget(self.example_button)
        header_layout.addWidget(self.open_button)
        outer.addWidget(header)

        body = QtWidgets.QHBoxLayout()
        body.setContentsMargins(16, 14, 16, 0)
        body.setSpacing(13)
        body.addWidget(self._build_sidebar())
        self.tabs = QtWidgets.QTabWidget()
        self.signal_tab = PlotTab('Signal comparison', 'signal')
        self.response_tab = PlotTab('Filter response', 'response')
        self.diagnostics_tab = PlotTab('Design diagnostics', 'diagnostics')
        self.plot_tabs = (self.signal_tab, self.response_tab, self.diagnostics_tab)
        for tab, label in zip(self.plot_tabs, ('Signals', 'Response', 'Diagnostics')):
            self.tabs.addTab(tab, label)
        self.tabs.addTab(self._build_code_tab(), 'Python')
        body.addWidget(self.tabs, 1)
        outer.addLayout(body, 1)

        footer = QtWidgets.QHBoxLayout()
        footer.setContentsMargins(17, 4, 17, 7)
        self.status_label = QtWidgets.QLabel('Open a signal file to get started.')
        self.status_label.setObjectName('footer')
        self.status_label.setTextFormat(QtCore.Qt.PlainText)
        self.status_label.setWordWrap(True)
        self.status_label.setFixedHeight(36)
        self.status_label.setSizePolicy(QtWidgets.QSizePolicy.Ignored, QtWidgets.QSizePolicy.Fixed)
        self.export_button = QtWidgets.QPushButton('Export CSV…')
        self.export_button.setProperty('role', 'tool')
        footer.addWidget(self.status_label, 1)
        footer.addWidget(self.export_button)
        outer.addLayout(footer)

    def _field(self, title: str, control: QtWidgets.QWidget) -> tuple[QtWidgets.QWidget, QtWidgets.QLabel]:
        wrapper = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(wrapper)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)
        label = QtWidgets.QLabel(title)
        label.setObjectName('fieldLabel')
        label.setBuddy(control)
        control.setAccessibleName(title)
        layout.addWidget(label)
        layout.addWidget(control)
        return wrapper, label

    def _build_sidebar(self) -> QtWidgets.QFrame:
        sidebar = QtWidgets.QFrame()
        sidebar.setObjectName('sidebar')
        sidebar.setFixedWidth(323)
        layout = QtWidgets.QVBoxLayout(sidebar)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(10)
        scroll = QtWidgets.QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        fields = QtWidgets.QWidget()
        fields.setObjectName('fields')
        form = QtWidgets.QVBoxLayout(fields)
        form.setContentsMargins(3, 3, 5, 4)
        form.setSpacing(6)
        data_title = QtWidgets.QLabel('DATA SOURCE')
        data_title.setObjectName('section')
        form.addWidget(data_title)
        self.file_label = QtWidgets.QLabel('No signal loaded')
        self.file_label.setObjectName('fileName')
        self.file_label.setTextFormat(QtCore.Qt.PlainText)
        self.file_label.setWordWrap(True)
        form.addWidget(self.file_label)
        self.input_combo = SettingComboBox()
        self.reference_combo = SettingComboBox((NO_REFERENCE,))
        self.rate_entry = QtWidgets.QLineEdit()
        self.rate_entry.setPlaceholderText('Enter Hz if not supplied')
        for label, control in (('Input channel', self.input_combo), ('Reference (optional)', self.reference_combo),
                               ('Sampling rate (Hz)', self.rate_entry)):
            widget, _ = self._field(label, control)
            form.addWidget(widget)
        self.rate_info = QtWidgets.QLabel()
        self.rate_info.setObjectName('secondary')
        self.rate_info.setWordWrap(True)
        form.addWidget(self.rate_info)
        divider = QtWidgets.QFrame()
        divider.setObjectName('divider')
        divider.setFrameShape(QtWidgets.QFrame.HLine)
        form.addWidget(divider)
        filter_title = QtWidgets.QLabel('FILTER SETTINGS')
        filter_title.setObjectName('section')
        form.addWidget(filter_title)

        self.type_combo = SettingComboBox(tuple(FILTER_TYPES))
        self.family_combo = SettingComboBox(tuple(FILTER_FAMILIES))
        self.mode_combo = SettingComboBox((CAUSAL, OFFLINE))
        self.order_entry = QtWidgets.QLineEdit('4')
        self.low_edge_entry = QtWidgets.QLineEdit('55')
        self.high_edge_entry = QtWidgets.QLineEdit('65')
        self.ripple_entry = QtWidgets.QLineEdit('1')
        self.attenuation_entry = QtWidgets.QLineEdit('40')
        self.type_combo.setCurrentText('Band-stop')
        grid = QtWidgets.QGridLayout()
        grid.setColumnStretch(0, 1)
        grid.setColumnStretch(1, 1)
        grid.setHorizontalSpacing(10)
        grid.setVerticalSpacing(8)
        for row, col, label, field in ((0, 0, 'Filter type', self.type_combo),
                                        (0, 1, 'Design family', self.family_combo),
                                        (1, 0, 'Prototype order', self.order_entry),
                                        (1, 1, 'Processing', self.mode_combo)):
            widget, _ = self._field(label, field)
            grid.addWidget(widget, row, col)
        self.low_field, self.low_heading = self._field('Low edge (Hz)', self.low_edge_entry)
        self.high_field, _ = self._field('High edge (Hz)', self.high_edge_entry)
        self.ripple_field, _ = self._field('Ripple (dB)', self.ripple_entry)
        self.attenuation_field, _ = self._field('Attenuation (dB)', self.attenuation_entry)
        for widget, row, col in ((self.low_field, 2, 0), (self.high_field, 2, 1),
                                 (self.ripple_field, 3, 0), (self.attenuation_field, 3, 1)):
            grid.addWidget(widget, row, col)
        form.addLayout(grid)
        self.edge_hint = QtWidgets.QLabel()
        self.edge_hint.setObjectName('secondary')
        self.edge_hint.setWordWrap(True)
        form.addWidget(self.edge_hint)

        # Additional frequency markers for the response plots
        marker_frequency_frame = QtWidgets.QWidget()
        marker_frequency_row = QtWidgets.QHBoxLayout(marker_frequency_frame)
        marker_frequency_row.setContentsMargins(0, 0, 0, 0)
        marker_frequency_row.setSpacing(4)
        self.marker_frequency_entries: list[QtWidgets.QLineEdit] = []
        for value in ('55', '65'):
            entry = QtWidgets.QLineEdit(value)
            marker_frequency_row.addWidget(entry)
            self.marker_frequency_entries.append(entry)
        self.marker_frequencies = self._marker_frequencies()
        marker_widget, _ = self._field('Marker frequencies (Hz) for testing your design', marker_frequency_frame)
        form.addWidget(marker_widget)
        form.addSpacing(10)
        form.addStretch(1)
        scroll.setWidget(fields)
        layout.addWidget(scroll, 1)
        return sidebar

    def _build_code_tab(self) -> QtWidgets.QWidget:
        tab = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(tab)
        layout.setContentsMargins(17, 15, 17, 13)
        heading = QtWidgets.QHBoxLayout()
        label = QtWidgets.QLabel('Reproduce this filter')
        label.setObjectName('plotTitle')
        self.copy_button = QtWidgets.QPushButton('Copy Python')
        self.copy_button.setProperty('role', 'tool')
        heading.addWidget(label, 1)
        heading.addWidget(self.copy_button)
        layout.addLayout(heading)
        self.code_text = QtWidgets.QPlainTextEdit()
        self.code_text.setReadOnly(True)
        self.code_text.setLineWrapMode(QtWidgets.QPlainTextEdit.NoWrap)
        self.code_text.setFont(QtGui.QFontDatabase.systemFont(QtGui.QFontDatabase.FixedFont))
        self.code_text.setPlainText('# SciPy code appears here after the filter is designed.')
        layout.addWidget(self.code_text, 1)
        return tab

    def _connect_controls(self) -> None:
        self.open_button.clicked.connect(self._open_file)
        self.example_button.clicked.connect(self._open_example)
        self.copy_button.clicked.connect(self._copy_code)
        self.export_button.clicked.connect(self._export)
        self.tabs.currentChanged.connect(self._schedule_plot)
        self.input_combo.currentIndexChanged.connect(self._channel_changed)
        for combo in (self.reference_combo, self.type_combo, self.family_combo, self.mode_combo):
            combo.currentIndexChanged.connect(self._selection_changed)
        for field in (self.rate_entry, self.order_entry, self.low_edge_entry, self.high_edge_entry,
                      self.ripple_entry, self.attenuation_entry):
            field.textEdited.connect(self._text_edited)
            field.editingFinished.connect(self._queue_filter_update)
        for field in self.marker_frequency_entries:
            field.editingFinished.connect(self._markers_changed)
        self.open_shortcut = QtWidgets.QShortcut(QtGui.QKeySequence(QtGui.QKeySequence.Open), self)
        self.open_shortcut.activated.connect(self._open_file)

    def eventFilter(self, watched: QtCore.QObject | None, event: QtCore.QEvent | None) -> bool:
        # Qt leaves focus in a line edit when the user clicks a passive label or background.
        if (event is not None and event.type() == QtCore.QEvent.MouseButtonPress
                and isinstance(watched, QtWidgets.QWidget) and watched.window() is self):
            field = QtWidgets.QApplication.focusWidget()
            if (isinstance(field, QtWidgets.QLineEdit) and field.window() is self
                    and watched is not field and not field.isAncestorOf(watched)):
                field.clearFocus()
        return super().eventFilter(watched, event)

    def _status(self, text: str, color: str = MUTED) -> None:
        self.status_label.setText(text)
        self.status_label.setToolTip(text)
        if self.status_label.property('statusColor') != color:
            self.status_label.setProperty('statusColor', color)
            self.status_label.setStyleSheet(f'color:{color};')

    def _sync_actions(self) -> None:
        kind = self._job.kind if self._job is not None else ''
        fresh = self.result is not None and self.result_revision == self.revision
        self.open_button.setEnabled(not self.busy)
        self.example_button.setEnabled(not self.busy)
        self.input_combo.setEnabled(self.source is not None and kind != 'load')
        self.reference_combo.setEnabled(self.reference_combo.count() > 1 and kind != 'load')
        self.export_button.setEnabled(fresh and not self.busy)
        self.copy_button.setEnabled(fresh and not self.busy)

    def _start_job(self, kind: str, operation: Callable[[], T], accept: Callable[[T], None],
                   status: str, error_prefix: str) -> None:
        if self.busy or self._closing:
            return
        future = self._worker.submit(operation)
        self._job = ActiveJob(kind, future, accept, error_prefix)
        self._status(status)
        self._sync_actions()
        future.add_done_callback(self._notify_finished)

    def _notify_finished(self, future: Future[Any]) -> None:
        if not self._closing:
            self._events.finished.emit(future)

    def _job_finished(self, future: Future[Any]) -> None:
        job = self._job
        if self._closing or job is None or job.future is not future:
            return
        self._job = None
        completed_filter = self._filter_in_flight
        if job.kind == 'filter':
            self._filter_in_flight = None
        if completed_filter is None or completed_filter.revision == self.revision:
            try:
                job.accept(future.result())
            except Exception as exc:
                self._status(f'{job.error_prefix}: {exc}', ERROR_COLOR)
        if self._job is None and self._pending_filter is not None:
            pending, self._pending_filter = self._pending_filter, None
            if pending.revision == self.revision:
                self._start_filter(pending)
        self._sync_actions()

    def _open_file(self) -> None:
        if self.busy:
            return
        folder = self.source.path.parent if self.source is not None and self.source.path is not None else ECG_FILE.parent
        patterns = ' '.join(f'*{extension}' for extension in SUPPORTED_EXTENSIONS)
        update_pending = self._update_timer.isActive()
        self._update_timer.stop()
        filename, _ = QtWidgets.QFileDialog.getOpenFileName(self, 'Open a signal file', str(folder),
                                                            f'Signal files ({patterns});;All files (*)')
        if filename:
            self._open_path(Path(filename))
        elif update_pending:
            self._queue_filter_update()

    def _open_path(self, path: Path) -> None:
        self._start_job('load', lambda: load_signal_file(path), self._set_source,
                        f'Opening {path.name}…', 'Could not open the file')

    def _open_example(self) -> None:
        if self.busy or self._closing:
            return
        if ECG_FILE.is_file():
            self._open_path(ECG_FILE)
        else:
            self._set_source(example_signal())

    def _set_source(self, source: SignalSource) -> None:
        self._update_timer.stop()
        self.result = None
        self._pending_filter = None
        self.rendered_tabs.clear()
        self.source = source
        self.channels = {channel.name: channel for channel in source.channels}
        chosen = ('ecg_noisy' if source.course_ecg and 'ecg_noisy' in self.channels
                  else next((c.name for c in source.channels if np.all(np.isfinite(c.samples))), source.channels[0].name))
        self._updating_form = True
        try:
            self.input_combo.clear()
            self.input_combo.addItems(list(self.channels))
            self.input_combo.setCurrentText(chosen)
            self.reference_combo.setCurrentText(NO_REFERENCE)
            self.file_label.setText(source.name)
            self.file_label.setToolTip(str(source.path) if source.path is not None else source.name)
            self._set_channel(prefer_reference=True)
            channel = self.channels[chosen]
            rate = channel.sample_rate_hz
            notch = (source.course_ecg or source.path is None) and rate is not None and math.isclose(rate, COURSE_RATE_HZ)
            self.family_combo.setCurrentText('Butterworth')
            self.type_combo.setCurrentText('Band-stop' if notch else 'Low-pass')
            self.order_entry.setText('4')
            self.mode_combo.setCurrentText(CAUSAL)
            self.low_edge_entry.setText('55' if notch else f'{rate / 10:.12g}' if rate is not None else '')
            self.high_edge_entry.setText('65' if notch else f'{rate / 5:.12g}' if rate is not None else '')
        finally:
            self._updating_form = False
        self._mark_settings_changed()
        if rate is not None:
            self._queue_filter_update()
        else:
            self.tabs.setCurrentIndex(0)
            self._status('Enter the sampling rate in Hz to start filtering.', WAIT_COLOR)
            self.rate_entry.setFocus()
        self._schedule_plot()

    def _channel_changed(self) -> None:
        if self._updating_form or self.input_combo.currentText() not in self.channels:
            return
        self._updating_form = True
        try:
            self._set_channel(prefer_reference=False)
        finally:
            self._updating_form = False
        self._selection_changed()
        if self.result is None:
            self.rendered_tabs.discard(0)
            self._schedule_plot()

    def _set_channel(self, *, prefer_reference: bool) -> None:
        channel = self.channels[self.input_combo.currentText()]
        options = (NO_REFERENCE, *(item.name for item in self.channels.values() if compatible_reference(channel, item)))
        previous = self.reference_combo.currentText()
        default = next((name for name in ('ecg_clean', 'Clean reference') if name in options), NO_REFERENCE)
        self.reference_combo.clear()
        self.reference_combo.addItems(options)
        self.reference_combo.setCurrentText(default if prefer_reference else previous if previous in options else NO_REFERENCE)
        self.rate_entry.setReadOnly(channel.time_seconds is not None)
        rate = channel.sample_rate_hz
        if rate is not None:
            self.rate_entry.setText(f'{rate:.12g}')
        elif prefer_reference or not self.rate_entry.isModified():
            self.rate_entry.clear()

    def _update_rate_info(self) -> None:
        channel = self.channels.get(self.input_combo.currentText())
        if channel is None:
            self.rate_info.setText('Rate from the file or entered manually')
            return
        try:
            rate = positive_number(float(self.rate_entry.text()), 'Sampling rate')
        except ValueError:
            self.rate_info.setText(f'{channel.samples.size:,} samples  ·  enter the rate in Hz')
            return
        origin = (channel.rate_source if channel.sample_rate_hz is not None
                  and math.isclose(channel.sample_rate_hz, rate, rel_tol=1e-11) else 'Manual rate')
        self.rate_info.setText(f'{channel.samples.size:,} samples  ·  Nyquist {rate / 2:g} Hz\n{origin or "Manual rate"}')

    def _update_filter_fields(self) -> None:
        band = self.type_combo.currentText() in {'Band-pass', 'Band-stop'}
        self.low_heading.setText('Low edge (Hz)' if band else 'Frequency (Hz)')
        self.low_edge_entry.setAccessibleName(self.low_heading.text())
        self.high_field.setVisible(band)
        family = FILTER_FAMILIES[self.family_combo.currentText()]
        self.ripple_field.setVisible(family in {'cheby1', 'ellip'})
        self.attenuation_field.setVisible(family in {'cheby2', 'ellip'})
        detail = EDGE_NAMES[family] + ' (single pass)'
        if band:
            try:
                detail += f'  ·  Order {2 * int(self.order_entry.text())} after band transform'
            except ValueError:
                pass
        mode = ('Zero phase uses forward/backward filtering.' if self.mode_combo.currentText() == OFFLINE
                else 'Causal processing uses one forward pass.')
        self.edge_hint.setText(detail + '\n' + mode)

    def _mark_settings_changed(self) -> None:
        self.revision += 1
        self._update_timer.stop()
        self._pending_filter = None
        self._update_filter_fields()
        self._update_rate_info()
        self._sync_actions()

    def _text_edited(self) -> None:
        self._mark_settings_changed()
        if self.source is not None and (self._job is None or self._job.kind != 'load'):
            self._status('Press Enter or leave the field to update.', WAIT_COLOR)

    def _selection_changed(self) -> None:
        if not self._updating_form:
            self._mark_settings_changed()
            self._queue_filter_update()

    def _queue_filter_update(self) -> None:
        if self._updating_form or self._closing or self.source is None or self.result_revision == self.revision:
            return
        if self._job is not None and self._job.kind == 'load':
            return
        self._status('Updating filter…')
        # A click can commit a text field and select a menu item in the same gesture.
        self._update_timer.start()

    def _suggest_initial_edges(self) -> None:
        if self.result is not None or self.low_edge_entry.text().strip() or self.high_edge_entry.text().strip():
            return
        try:
            rate = positive_number(float(self.rate_entry.text()), 'Sampling rate')
        except ValueError:
            return
        self.low_edge_entry.setText(f'{rate / 10:.12g}')
        self.high_edge_entry.setText(f'{rate / 5:.12g}')

    def _marker_frequencies(self) -> tuple[float, ...]:
        """Parse the marker fields. Blank fields are ignored."""
        frequencies: list[float] = []
        for entry in self.marker_frequency_entries:
            text = entry.text().strip()
            if not text:
                continue
            try:
                frequency = float(text)
            except ValueError as exc:
                raise SignalError(f'Enter a number for marker frequency "{text}", or leave it blank.') from exc
            if not math.isfinite(frequency) or frequency <= 0:
                raise SignalError('Marker frequencies must be positive numbers in Hz.')
            frequencies.append(frequency)
        return tuple(frequencies)

    def _markers_changed(self) -> None:
        try:
            markers = self._marker_frequencies()
        except SignalError as exc:
            self._status(f'{exc} Showing the previous markers.', ERROR_COLOR)
            return
        if markers == self.marker_frequencies:
            return
        self.marker_frequencies = markers
        # Only the frequency-axis plots show markers; the filter itself is unchanged.
        self.rendered_tabs.difference_update({0, 1})
        self._schedule_plot()

    def _settings(self) -> FilterSettings:
        def number(field: QtWidgets.QLineEdit, label: str) -> float:
            try:
                return float(field.text())
            except ValueError as exc:
                raise SignalError(f'Enter a number for {label}.') from exc

        channel = self.channels[self.input_combo.currentText()]
        rate = number(self.rate_entry, 'sampling rate in Hz')
        if channel.sample_rate_hz is not None and self.rate_entry.text() == f'{channel.sample_rate_hz:.12g}':
            rate = channel.sample_rate_hz
        try:
            order = int(self.order_entry.text())
        except ValueError as exc:
            raise SignalError('Enter a whole number for filter order.') from exc
        kind = FILTER_TYPES[self.type_combo.currentText()]
        family = FILTER_FAMILIES[self.family_combo.currentText()]
        edges: tuple[float, ...] = (number(self.low_edge_entry, 'the frequency'),)
        if kind in {'bandpass', 'bandstop'}:
            edges += (number(self.high_edge_entry, 'the high edge'),)
        ripple = number(self.ripple_entry, 'ripple') if family in {'cheby1', 'ellip'} else None
        attenuation = number(self.attenuation_entry, 'attenuation') if family in {'cheby2', 'ellip'} else None
        settings = FilterSettings(rate, order, kind, family, edges, self.mode_combo.currentText() == OFFLINE,
                                  ripple, attenuation)
        validate_settings(settings)
        return settings

    def _update_filter(self) -> None:
        if self.source is None or self._closing:
            return
        if self._job is not None and self._job.kind == 'load':
            return
        if self.result_revision == self.revision:
            return
        if not self.rate_entry.text().strip():
            self._status('Enter the sampling rate in Hz to start filtering.', WAIT_COLOR)
            return
        self._suggest_initial_edges()
        try:
            settings = self._settings()
        except SignalError as exc:
            previous = ' Showing the previous result.' if self.result is not None else ''
            self._status(f'{exc}{previous}', ERROR_COLOR)
            return
        request = FilterRequest(self.revision, settings, self.source.name,
                                self.channels[self.input_combo.currentText()],
                                self.channels.get(self.reference_combo.currentText()),
                                f'{self.family_combo.currentText()} {self.type_combo.currentText().casefold()}')
        if self.busy:
            if self._filter_in_flight is None or self._filter_in_flight.revision != request.revision:
                self._pending_filter = request
                self._status('Latest settings queued…', WAIT_COLOR)
        else:
            self._start_filter(request)

    def _start_filter(self, request: FilterRequest) -> None:
        self._filter_in_flight = request
        self._start_job('filter', request.calculate, lambda result: self._show_result(request, result),
                        'Updating filter…', 'Could not apply the filter')

    def _show_result(self, request: FilterRequest, result: FilterResult) -> None:
        self.result = result
        self.result_revision = request.revision
        self.rendered_tabs.clear()
        mode = OFFLINE if request.settings.zero_phase else CAUSAL
        self._status(f'{request.description}  ·  order {len(result.design.poles)}  ·  {mode}  ·  '
                     f'{result.output.size:,} samples filtered')
        self._sync_actions()
        self._schedule_plot()

    def _schedule_plot(self) -> None:
        # Let the tab/button paint first. Coalesce rapid switches and draw only
        # the selected panel, once per applied result.
        self._plot_timer.stop()
        index = self.tabs.currentIndex()
        if index >= 0 and index not in self.rendered_tabs:
            if self.result is not None and index < len(self.plot_tabs):
                self.plot_tabs[index].subtitle.setText('Updating plot…')
            self._plot_timer.start()

    def _draw_current_tab(self) -> None:
        index = self.tabs.currentIndex()
        if self._closing or index < 0 or index in self.rendered_tabs:
            return
        result = self.result
        if result is None:
            if index == 3:
                self.code_text.setPlainText('# SciPy code appears here after the filter is designed.')
            elif index == 0 and self.source is not None and not self.rate_entry.text().strip():
                draw_input_preview(self.signal_tab, self.channels[self.input_combo.currentText()])
            elif self.source is not None:
                self.plot_tabs[index].show_empty('Results appear once the filter settings are valid', self.source.name)
            else:
                self.plot_tabs[index].show_empty('Open a signal file to begin', 'No signal loaded')
        elif index == 0:
            draw_signal_comparison(self.signal_tab, result, self.marker_frequencies)
        elif index == 1:
            draw_filter_response(self.response_tab, result.design, self.marker_frequencies)
        elif index == 2:
            draw_filter_diagnostics(self.diagnostics_tab, result.design)
        else:
            self.code_text.setPlainText(python_example(result.design.settings))
        self.rendered_tabs.add(index)

    def _copy_code(self) -> None:
        if self.result is not None and self.result_revision == self.revision:
            clipboard = QtWidgets.QApplication.clipboard()
            if clipboard is not None:
                clipboard.setText(python_example(self.result.design.settings))
                self._status('Python code copied to the clipboard.')

    def _export(self) -> None:
        result = self.result
        if result is None or self.result_revision != self.revision or self.busy:
            return
        filename, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, 'Export filtered samples', f'{Path(result.source_name).stem}_filtered.csv', 'CSV files (*.csv)')
        if filename:
            path = Path(filename)
            if not path.suffix:
                path = path.with_suffix('.csv')
            self._start_job('export', lambda: export_signal_csv(path, result),
                            lambda _: self._status(f'Saved {path.name}.'), 'Exporting samples…', 'Could not export the CSV')

    def closeEvent(self, event: QtGui.QCloseEvent) -> None:
        self._closing = True
        application = QtWidgets.QApplication.instance()
        if application is not None:
            application.removeEventFilter(self)
        self._update_timer.stop()
        self._plot_timer.stop()
        self._pending_filter = None
        self._worker.shutdown(wait=False, cancel_futures=True)
        super().closeEvent(event)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('file', nargs='?', type=Path, help='optional signal file to open at startup')
    args = parser.parse_args()
    application = QtWidgets.QApplication([sys.argv[0]])
    application.setStyle('Fusion')
    window = FilterDesignTool(args.file.expanduser().resolve() if args.file is not None else None)
    window.show()
    sys.exit(application.exec_())


if __name__ == '__main__':
    main()
