#!/usr/bin/env python3

import numpy as np
import matplotlib
matplotlib.use('Agg')  # 使用非交互式后端
import matplotlib.pyplot as plt
from matplotlib.widgets import Cursor
from scipy.signal import freqz
import ctypes
import os

print("Testing without GUI...")

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
so_file = "iir1_filter.so"
with open(c_file, "w") as f:
    f.write(c_code)

os.system(f"gcc -shared -fPIC -o {so_file} {c_file}")

# Load the shared library
lib = ctypes.CDLL(f"./{so_file}")

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

print(f"Filter coefficients: a={a_coeff:.6f}, b={b_coeff:.6f}")
print(f"Frequency range: {freq_min} Hz to {freq_max} Hz")
print(f"Number of frequency points: {len(freq_plot)}")
print(f"Amplitude range: {amp_db.min():.2f} dB to {amp_db.max():.2f} dB")

# Test C implementation
def test_c_filter():
    filt = IIR1()
    lib.iir1_init(ctypes.byref(filt), a_coeff, b_coeff)
    
    # Test with a simple step input
    inputs = [0, 1, 1, 1, 1, 1]
    outputs = []
    
    for inp in inputs:
        out = lib.iir1_step(ctypes.byref(filt), inp)
        outputs.append(out)
    
    print("C filter test:")
    print(f"  Inputs:  {inputs}")
    print(f"  Outputs: {[f'{x:.6f}' for x in outputs]}")
    
test_c_filter()

# Create a simple plot and save it
plt.figure(figsize=(12, 8))
plt.subplot(2, 1, 1)
plt.semilogx(freq_plot, amp_db)
plt.grid(True)
plt.xlabel('Frequency [Hz]')
plt.ylabel('Amplitude [dB]')
plt.title('RC Filter Frequency Response')

plt.subplot(2, 1, 2)
plt.semilogx(freq_plot, angles)
plt.grid(True)
plt.xlabel('Frequency [Hz]')
plt.ylabel('Phase [rad]')

plt.tight_layout()
plt.savefig('filter_response.png', dpi=150, bbox_inches='tight')
print("Plot saved as 'filter_response.png'")

print("Test completed successfully!")