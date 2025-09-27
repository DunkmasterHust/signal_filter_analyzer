# 1st Order RC Lowpass Filter Frequency Response Simulation
# --------------------------------------------------------
# This script simulates and compares the frequency response of a 1st order RC lowpass filter using both Python (scipy) and a C implementation via ctypes.
#
# Filter Design Mechanism:
# ------------------------
# The analog RC lowpass filter has the transfer function:
#     H(s) = 1 / (1 + sRC)
# where R is resistance, C is capacitance, and s is the Laplace variable.
#
# To implement this in discrete time (digital), we use the difference equation:
#     y[n] = a * y[n-1] + b * x[n]
# where:
#     a = exp(-dt/RC)
#     b = 1 - a
#     dt = 1/fs (fs is the sampling frequency)
#
# This is derived by matching the continuous-time and discrete-time system responses at the sampling instants (impulse-invariant or matched-z transform method).
# The resulting filter is a simple exponential smoothing filter, which closely approximates the analog RC filter for fs >> 1/(2πRC).
#
# The script computes the frequency response for both the Python/scipy and C implementations, and displays amplitude and phase from 1Hz to 1kHz.
# The plot includes:
#   - Interactive movable x/y cursors for measuring amplitude and phase at any frequency
#   - The -3dB cutoff frequency and phase margin at the -3dB point
#
# The C code is compiled and loaded at runtime for direct comparison.

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Cursor
from matplotlib.widgets import Cursor
from scipy.signal import freqz
import ctypes
import os
import platform

# Write the C code for a 1st order RC filter: Yn = a*Yn-1 + b*Xn
c_code = """
#include <stddef.h>

typedef struct {
    double a;
    double b;
    double y_prev;
} IIR1;

void iir1_init(IIR1* f, double a, double b) {
    f->a = a;
    f->b = b;
    f->y_prev = 0.0;
}

double iir1_step(IIR1* f, double input) {
    double output = f->a * f->y_prev + f->b * input;
    f->y_prev = output;
    return output;
}
"""

# Save and compile the C code as a shared library
c_file = "iir1_filter.c"

# Select appropriate file extension and compilation options based on operating system
if platform.system() == "Windows":
    lib_file = "iir1_filter.dll"
    compile_cmd = f"gcc -shared -o {lib_file} {c_file}"
else:
    lib_file = "iir1_filter.so"
    compile_cmd = f"gcc -shared -fPIC -o {lib_file} {c_file}"

# Write C code file
print(f"Writing C code to {c_file}...")
with open(c_file, "w") as f:
    f.write(c_code.strip())  # Remove leading and trailing whitespace

# Compile C code
print(f"Compiling with command: {compile_cmd}")
compile_result = os.system(compile_cmd)
print(f"Compilation result: {compile_result}")

if compile_result != 0:
    print(f"Compilation failed. Checking if output file exists: {os.path.exists(lib_file)}")
    # Try to get more detailed error information
    import subprocess
    try:
        result = subprocess.run(compile_cmd.split(), capture_output=True, text=True, cwd='.')
        print(f"Detailed error output:")
        print(f"STDOUT: {result.stdout}")
        print(f"STDERR: {result.stderr}")
    except Exception as e:
        print(f"Failed to get detailed error: {e}")
    
    # If file exists, compilation may have succeeded but returned non-zero exit code
    if not os.path.exists(lib_file):
        raise RuntimeError(f"C code compilation failed with exit code: {compile_result}")
    else:
        print("Warning: Compilation returned non-zero exit code but output file exists. Continuing...")

# Load shared library
print(f"Loading shared library: {lib_file}")
try:
    lib = ctypes.CDLL(f"./{lib_file}")
    print("Library loaded successfully!")
except OSError as e:
    print(f"Failed to load library {lib_file}: {e}")
    print(f"Current working directory: {os.getcwd()}")
    print(f"Library file exists: {os.path.exists(lib_file)}")
    if os.path.exists(lib_file):
        import stat
        file_stat = os.stat(lib_file)
        print(f"Library file size: {file_stat.st_size} bytes")
        print(f"Library file permissions: {oct(file_stat.st_mode)}")
    raise RuntimeError(f"Cannot load shared library {lib_file}: {e}")

# Define ctypes structures and functions
class IIR1(ctypes.Structure):
    _fields_ = [
        ("a", ctypes.c_double),
        ("b", ctypes.c_double),
        ("y_prev", ctypes.c_double),
    ]


lib.iir1_init.argtypes = [ctypes.POINTER(IIR1), ctypes.c_double, ctypes.c_double]
lib.iir1_step.argtypes = [ctypes.POINTER(IIR1), ctypes.c_double]
lib.iir1_step.restype = ctypes.c_double


# Frequency response using scipy (reference)
fs = 20000  # Higher sample rate for more ideal digital RC filter
# RC filter coefficients for digital implementation:
#   y[n] = a*y[n-1] + b*x[n]
#   a = exp(-dt/RC), b = 1-a
#   dt = 1/fs
#   RC sets the cutoff frequency (fc = 1/(2πRC))
#   For 10 Hz cutoff: RC = 1/(2*pi*10)
fc = 10  # cutoff frequency in Hz for both analog and digital
R = 1e3  # 1k ohm
C = 1/(2*np.pi*fc*R)  # C in Farads
RC = R * C
dt = 1/float(fs)
a_coeff = np.exp(-dt/RC)
b_coeff = 1 - a_coeff

# Frequency response using scipy (reference)
w, h = freqz([b_coeff], [1, -a_coeff], worN=16000)
freq = w * fs / (2 * np.pi)
freq_min = 0.1
freq_max = 40000
mask = (freq >= freq_min) & (freq <= freq_max)
freq_plot = freq[mask]
amp_db = 20 * np.log10(np.abs(h[mask]))
angles = np.angle(h[mask])

# Calculate ideal analog RC filter frequency response
def analog_rc_freq_response(RC, f):
    w = 2 * np.pi * f
    H = 1 / (1 + 1j * w * RC)
    return H

freq_analog = np.logspace(np.log10(freq_min), np.log10(freq_max), 2000)
H_analog = analog_rc_freq_response(RC, freq_analog)
amp_db_analog = 20 * np.log10(np.abs(H_analog))
angles_analog = np.angle(H_analog)

def c_freq_response(a_coeff, b_coeff, fs, n_points=2000):
    N_total = 4096
    N_steady = 1024
    f = np.linspace(0.1, 40000, n_points)  # 0.1Hz to 40kHz
    amp = np.zeros(n_points)
    phase = np.zeros(n_points)
    for i, freq_hz in enumerate(f):
        if freq_hz == 0:
            w = 0.0
        else:
            w = 2 * np.pi * freq_hz / fs
        n = np.arange(N_total)
        x_real = np.cos(w * n)
        x_imag = np.sin(w * n)
        y_real = np.zeros(N_total)
        y_imag = np.zeros(N_total)
        # Real part
        filt = IIR1()
        lib.iir1_init(ctypes.byref(filt), a_coeff, b_coeff)
        for k in range(N_total):
            y_real[k] = lib.iir1_step(ctypes.byref(filt), x_real[k])
        # Imag part
        filt = IIR1()
        lib.iir1_init(ctypes.byref(filt), a_coeff, b_coeff)
        for k in range(N_total):
            y_imag[k] = lib.iir1_step(ctypes.byref(filt), x_imag[k])
        y = y_real + 1j * y_imag
        x = x_real + 1j * x_imag
        H = np.mean(y[-N_steady:] / x[-N_steady:])
        amp[i] = np.abs(H)
        phase[i] = np.angle(H)
    return f, amp, phase

f_c, amp_c, phase_c = c_freq_response(a_coeff, b_coeff, fs, n_points=2000)
mask_c = (f_c >= freq_min) & (f_c <= freq_max)

# Find -3dB cutoff frequency (Python/scipy)
amp_db_full = 20 * np.log10(np.abs(h))
idx_3db = np.argmin(np.abs(amp_db_full - (amp_db_full[0] - 3)))
f_3db = freq[idx_3db]
amp_3db = amp_db_full[idx_3db]
phase_3db = np.angle(h[idx_3db])
phase_margin_3db = 180 + np.angle(h[idx_3db], deg=True)

# Plot amplitude and phase response
fig, ax1 = plt.subplots(2, 1, figsize=(8, 6))

# Amplitude response (in dB)
ax1[0].plot(freq_plot, amp_db, 'b', label='Python (scipy)')
ax1[0].plot(f_c[mask_c], 20 * np.log10(amp_c[mask_c]), 'r--', label='C module')
ax1[0].plot(freq_analog, amp_db_analog, 'k-.', label='Analog RC (ideal)')
ax1[0].set_title('1st Order IIR Filter Frequency Response (Python & C)')
ax1[0].set_ylabel('Amplitude [dB]')
ax1[0].set_xlim(freq_min, freq_max)
ax1[0].set_xscale('log')
ax1[0].grid()
ax1[0].legend()
# Mark -3dB point
ax1[0].plot(f_3db, amp_3db, 'ko')
ax1[0].annotate(f'-3dB @ {f_3db:.1f}Hz', xy=(f_3db, amp_3db), xytext=(f_3db+20, amp_3db-10),
                arrowprops=dict(facecolor='black', shrink=0.05), fontsize=9)

# Phase response
ax1[1].plot(freq_plot, angles, 'g', label='Python (scipy)')
ax1[1].plot(f_c[mask_c], phase_c[mask_c], 'm--', label='C module')
ax1[1].plot(freq_analog, angles_analog, 'k-.', label='Analog RC (ideal)')
ax1[1].set_ylabel('Phase [radians]')
ax1[1].set_xlabel('Frequency [Hz]')
ax1[1].set_xlim(freq_min, freq_max)
ax1[1].set_xscale('log')
ax1[1].grid()
ax1[1].legend()
# Mark phase margin at -3dB point
ax1[1].plot(f_3db, phase_3db, 'ko')
ax1[1].annotate(f'Phase Margin: {phase_margin_3db:.1f}° @ {f_3db:.1f}Hz',
                xy=(f_3db, phase_3db),
                xytext=(f_3db+20, phase_3db+0.5),
                arrowprops=dict(facecolor='black', shrink=0.05), fontsize=9)


# Add interactive movable x/y cursors for both subplots
cursor0 = Cursor(ax1[0], useblit=True, color='gray', linewidth=1, linestyle='--')
cursor1 = Cursor(ax1[1], useblit=True, color='gray', linewidth=1, linestyle='--')

# Display coordinates for both Python and C module curves on mouse move
def format_coord_both_curves(x, y, x_py, y_py, x_c, y_c, label):
    # Find nearest index for Python curve
    idx_py = np.searchsorted(x_py, x)
    if idx_py >= len(x_py):
        idx_py = len(x_py) - 1
    # Find nearest index for C module curve
    idx_c = np.searchsorted(x_c, x)
    if idx_c >= len(x_c):
        idx_c = len(x_c) - 1
    return (f"{label}: freq={x:.2f} Hz | "
            f"Python={y_py[idx_py]:.2f}, C module={y_c[idx_c]:.2f}")

# Amplitude subplot: show both Python and C module dB values
def amp_format_coord(x, y):
    return format_coord_both_curves(
        x, y,
        freq_plot, amp_db,
        f_c[mask_c], 20 * np.log10(amp_c[mask_c]),
        'Amplitude [dB]'
    )

# Phase subplot: show both Python and C module phase values
def phase_format_coord(x, y):
    return format_coord_both_curves(
        x, y,
        freq_plot, angles,
        f_c[mask_c], phase_c[mask_c],
        'Phase [rad]'
    )

ax1[0].format_coord = amp_format_coord
ax1[1].format_coord = phase_format_coord

plt.tight_layout()

# plt.show() should remain to display the main plots
plt.show()

# -----------------------------------------------------------------------------
# Filter Difference Function (for reference)
# -----------------------------------------------------------------------------
#
# Reference: 1st order RC filter (discrete): y[n] = a*y[n-1] + b*x[n]
# a = exp(-dt/RC), b = 1-a
#

# -----------------------------------------------------------------------------
# How to Calculate 'a' and 'b' Coefficients for the 1st Order RC Digital Filter
# -----------------------------------------------------------------------------
# Derivation Process:
#
# 1. Continuous-time transfer function of analog RC lowpass filter:
#      H(s) = 1 / (1 + sRC)
#
# 2. Corresponding differential equation:
#      v_out(t) + RC * dv_out/dt = v_in(t)
#
# 3. Discretization (using sampling period dt=1/fs, forward/backward Euler method, or zero-order hold/matched z-transform):
#    Here we use "equivalent first-order inertial discretization" or "matched z-transform":
#
#    Let y[n] ≈ v_out(n*dt), x[n] ≈ v_in(n*dt)
#
#    Discrete recursive formula:
#      y[n] = a * y[n-1] + b * x[n]
#
#    Where:
#      a = exp(-dt/RC)
#      b = 1 - a
#
#    Brief derivation explanation:
#      - Unit impulse response of RC circuit: h(t) = (1/RC) * exp(-t/RC) * u(t)
#      - After discrete sampling: h[n] = (1/RC) * exp(-n*dt/RC) * dt
#      - Z-domain transfer function of discrete system: H(z) = b / (1 - a*z^-1)
#      - Match discrete system poles with continuous system poles at sampling points (matched z-transform): a = exp(-dt/RC)
#      - b determined by unit step response converging to 1: b = 1 - a
#
# 4. Calculation steps:
#      - Given sampling rate fs (Hz)
#      - Given cutoff frequency fc (Hz)
#      - RC = 1 / (2 * pi * fc)
#      - dt = 1 / fs
#      - a = exp(-dt / RC)
#      - b = 1 - a
#
# 5. Code example:
#      fs = 20000         # Sampling rate (Hz)
#      fc = 10            # Cutoff frequency (Hz)
#      dt = 1 / fs
#      RC = 1 / (2 * np.pi * fc)
#      a = np.exp(-dt / RC)
#      b = 1 - a
#
# 6. Discrete filter difference equation:
#      y[n] = a * y[n-1] + b * x[n]
#
# -----------------------------------------------------------------------------
