import numpy as np
import pyqtgraph as pg
from pyqtgraph.Qt import QtWidgets


def generate_signal(fs=2000, base_freq=1.0, duration=10.0, noise_level=0.2, seed=42):
    rng = np.random.default_rng(seed)
    t = np.arange(0, duration, 1.0 / fs)
    fundamental = np.sin(2 * np.pi * base_freq * t)
    harmonics = [
        0.3 * np.sin(2 * np.pi * 2 * base_freq * t + 0.1),
        0.2 * np.sin(2 * np.pi * 3 * base_freq * t + 0.5),
        0.15 * np.sin(2 * np.pi * 5 * base_freq * t - 0.3),
        0.1 * np.sin(2 * np.pi * 50 * t + 1.0),
        0.05 * np.sin(2 * np.pi * 120 * t + 2.3),
    ]
    broadband = rng.normal(0, noise_level, size=t.shape)
    low_freq_drift = 0.1 * np.sin(2 * np.pi * 0.2 * t)
    signal = fundamental + sum(harmonics) + broadband + low_freq_drift
    return t, signal


def export_csv(path, time, signal):
    data = np.column_stack((time, signal))
    np.savetxt(path, data, delimiter=",", header="time,amplitude", comments="")


def compute_spectrum(signal, fs):
    fft_vals = np.fft.rfft(signal)
    freqs = np.fft.rfftfreq(len(signal), 1.0 / fs)
    magnitude = np.abs(fft_vals) * 2.0 / len(signal)
    return freqs, magnitude


class Crosshair:
    def __init__(self, plot_item, fmt):
        pen = pg.mkPen(color="#ffa500", width=1)
        self.plot_item = plot_item
        self.formatter = fmt
        self._updating = False

        self.v_line = pg.InfiniteLine(angle=90, movable=True, pen=pen)
        self.h_line = pg.InfiniteLine(angle=0, movable=True, pen=pen)
        self.text = pg.TextItem(anchor=(1, 1))

        plot_item.addItem(self.v_line, ignoreBounds=True)
        plot_item.addItem(self.h_line, ignoreBounds=True)
        plot_item.addItem(self.text)

        self.v_line.sigPositionChanged.connect(self._line_moved)
        self.h_line.sigPositionChanged.connect(self._line_moved)

        self.proxy = pg.SignalProxy(
            plot_item.scene().sigMouseMoved,
            rateLimit=60,
            slot=self._mouse_moved,
        )

    def _mouse_moved(self, event):
        if not event:
            return
        pos = event[0]
        if not self.plot_item.sceneBoundingRect().contains(pos):
            return
        vb = self.plot_item.getViewBox()
        if vb is None:
            return
        mouse_point = vb.mapSceneToView(pos)
        self._set_position(mouse_point.x(), mouse_point.y())

    def _line_moved(self, _):
        if self._updating:
            return
        self._update_text(self.v_line.value(), self.h_line.value())

    def _set_position(self, x, y):
        self._updating = True
        self.v_line.setPos(x)
        self.h_line.setPos(y)
        self._updating = False
        self._update_text(x, y)

    def _update_text(self, x, y):
        self.text.setHtml(self.formatter.format(x=x, y=y))
        self.text.setPos(x, y)


def main():
    fs = 2000
    duration = 10.0
    csv_path = "harmonic_signal.csv"

    time, signal = generate_signal(fs=fs, duration=duration)
    export_csv(csv_path, time, signal)

    freqs, magnitude = compute_spectrum(signal, fs)

    app = pg.mkQApp("Harmonic Signal Viewer")
    window = QtWidgets.QMainWindow()
    central = QtWidgets.QWidget()
    layout = QtWidgets.QVBoxLayout(central)

    time_plot = pg.PlotWidget(title="Time Domain")
    time_plot.plot(time, signal, pen=pg.mkPen(color="#0077cc", width=1))
    time_plot.setLabel("bottom", "Time", units="s")
    time_plot.setLabel("left", "Amplitude")

    freq_plot = pg.PlotWidget(title="Frequency Domain")
    freq_plot.plot(freqs, magnitude, pen=pg.mkPen(color="#cc3300", width=1))
    freq_plot.setLabel("bottom", "Frequency", units="Hz")
    freq_plot.setLabel("left", "Magnitude")
    freq_plot.setXRange(0, 200)

    layout.addWidget(time_plot)
    layout.addWidget(freq_plot)

    window.setCentralWidget(central)
    window.resize(900, 700)
    window.show()

    Crosshair(time_plot.getPlotItem(), "<span>t = {x:.3f} s<br>amp = {y:.3f}</span>")
    Crosshair(freq_plot.getPlotItem(), "<span>f = {x:.1f} Hz<br>|X| = {y:.3f}</span>")

    app.exec()


if __name__ == "__main__":
    main()
