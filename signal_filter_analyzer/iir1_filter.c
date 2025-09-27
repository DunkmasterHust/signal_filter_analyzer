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