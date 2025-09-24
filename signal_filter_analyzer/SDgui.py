import numpy as np
import csv
import sys
from PySide6.QtWidgets import QApplication, QWidget, QGridLayout
import pyqtgraph as pg

# 1. 生成随机数据
data1 = np.random.randn(1024)
data2 = np.random.randn(4096)
data3 = np.random.randn(51200)

# 2. 写入CSV文件
csv_file = 'random_data.csv'
with open(csv_file, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['data1', 'data2', 'data3'])
    for i in range(max(len(data1), len(data2), len(data3))):
        row = [
            data1[i] if i < len(data1) else '',
            data2[i] if i < len(data2) else '',
            data3[i] if i < len(data3) else ''
        ]
        writer.writerow(row)

# 3. 读取CSV文件
read_data1, read_data2, read_data3 = [], [], []
with open(csv_file, 'r') as f:
    reader = csv.DictReader(f)
    for row in reader:
        if row['data1']:
            read_data1.append(float(row['data1']))
        if row['data2']:
            read_data2.append(float(row['data2']))
        if row['data3']:
            read_data3.append(float(row['data3']))
read_data1 = np.array(read_data1)
read_data2 = np.array(read_data2)
read_data3 = np.array(read_data3)

# 4. PyQtGraph显示
class MainWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Random Data Analysis")
        layout = QGridLayout(self)

        # 正态分布（直方图）
        hist1 = pg.PlotWidget(title="Data1 Histogram")
        y, x = np.histogram(read_data1, bins=50)
        hist1.plot(x[:-1], y, stepMode=False, fillLevel=0, brush=(150,150,255,150))
        layout.addWidget(hist1, 0, 0)

        hist2 = pg.PlotWidget(title="Data2 Histogram")
        y, x = np.histogram(read_data2, bins=50)
        hist2.plot(x[:-1], y, stepMode=False, fillLevel=0, brush=(150,150,255,150))
        layout.addWidget(hist2, 0, 1)

        hist3 = pg.PlotWidget(title="Data3 Histogram")
        y, x = np.histogram(read_data3, bins=50)
        hist3.plot(x[:-1], y, stepMode=False, fillLevel=0, brush=(150,150,255,150))
        layout.addWidget(hist3, 0, 2)

        # FFT频率图
        fft1 = np.abs(np.fft.fft(read_data1))
        freq1 = np.fft.fftfreq(len(read_data1))
        fft_plot1 = pg.PlotWidget(title="Data1 FFT")
        fft_plot1.plot(freq1, fft1)
        layout.addWidget(fft_plot1, 1, 0)

        fft2 = np.abs(np.fft.fft(read_data2))
        freq2 = np.fft.fftfreq(len(read_data2))
        fft_plot2 = pg.PlotWidget(title="Data2 FFT")
        fft_plot2.plot(freq2, fft2)
        layout.addWidget(fft_plot2, 1, 1)

        fft3 = np.abs(np.fft.fft(read_data3))
        freq3 = np.fft.fftfreq(len(read_data3))
        fft_plot3 = pg.PlotWidget(title="Data3 FFT")
        fft_plot3.plot(freq3, fft3)
        layout.addWidget(fft_plot3, 1, 2)

if __name__ == '__main__':
    app = QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec())