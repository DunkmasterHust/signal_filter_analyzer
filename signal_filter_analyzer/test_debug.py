#!/usr/bin/env python3

# 分步测试脚本
import sys
print("Step 1: Starting script")

try:
    import numpy as np
    print("Step 2: numpy imported successfully")
except Exception as e:
    print(f"Error importing numpy: {e}")
    sys.exit(1)

try:
    import matplotlib.pyplot as plt
    from matplotlib.widgets import Cursor
    print("Step 3: matplotlib imported successfully")
except Exception as e:
    print(f"Error importing matplotlib: {e}")
    sys.exit(1)

try:
    from scipy.signal import freqz
    print("Step 4: scipy imported successfully")
except Exception as e:
    print(f"Error importing scipy: {e}")
    sys.exit(1)

try:
    import ctypes
    import os
    print("Step 5: ctypes and os imported successfully")
except Exception as e:
    print(f"Error importing ctypes/os: {e}")
    sys.exit(1)

# 检查C代码编译
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

try:
    c_file = "iir1_filter.c"
    so_file = "iir1_filter.so"
    with open(c_file, "w") as f:
        f.write(c_code)
    print("Step 6: C code written successfully")
    
    # 编译
    result = os.system(f"gcc -shared -fPIC -o {so_file} {c_file}")
    if result == 0:
        print("Step 7: C code compiled successfully")
    else:
        print(f"Step 7: Compilation failed with code {result}")
        sys.exit(1)
        
    # 加载库
    lib = ctypes.CDLL(f"./{so_file}")
    print("Step 8: Library loaded successfully")
    
except Exception as e:
    print(f"Error in compilation/loading: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# 定义ctypes结构
try:
    class IIR1(ctypes.Structure):
        _fields_ = [
            ("a", ctypes.c_double),
            ("b", ctypes.c_double),
            ("y_prev", ctypes.c_double),
        ]

    lib.iir1_init.argtypes = [ctypes.POINTER(IIR1), ctypes.c_double, ctypes.c_double]
    lib.iir1_step.argtypes = [ctypes.POINTER(IIR1), ctypes.c_double]
    lib.iir1_step.restype = ctypes.c_double
    print("Step 9: ctypes structures defined successfully")
except Exception as e:
    print(f"Error defining ctypes structures: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# 测试基本计算
try:
    fs = 20000
    fc = 10
    R = 1e3
    C = 1/(2*np.pi*fc*R)
    RC = R * C
    dt = 1/float(fs)
    a_coeff = np.exp(-dt/RC)
    b_coeff = 1 - a_coeff
    print(f"Step 10: Coefficients calculated: a={a_coeff:.6f}, b={b_coeff:.6f}")
except Exception as e:
    print(f"Error calculating coefficients: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# 测试频率响应计算
try:
    w, h = freqz([b_coeff], [1, -a_coeff], worN=16000)
    freq = w * fs / (2 * np.pi)
    print("Step 11: Frequency response calculated successfully")
except Exception as e:
    print(f"Error calculating frequency response: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("All steps completed successfully!")