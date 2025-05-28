import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from sklearn.ensemble import IsolationForest
from matplotlib.colors import ListedColormap
import librosa  # For audio processing
import os

# Load your machine audio file
# audio_path = os.path.abspath(r"C:\Users\goaic\Desktop\plot_fft\audio\machine_sound.wav")
# y, sr = librosa.load(audio_path, sr=None)  # y=audio samples, sr=sampling rate
# TOTAL_POINTS = len(y)  # Use all audio samples

# 参数设定
np.random.seed(42)
TOTAL_POINTS = 500
ANIMATION_SPEED = 50

# 生成时间序列数据
t = np.linspace(0, 10, TOTAL_POINTS)
base_signal = np.sin(2 * np.pi * t) + 0.1 * t
y = base_signal + np.random.normal(0, 0.5, TOTAL_POINTS)

# 添加人工异常点
anomaly_indices = np.random.choice(TOTAL_POINTS, 20, replace=False)
y[anomaly_indices] += np.random.uniform(-7, 7, 20)

# 创建二维特征 (时间延迟嵌入)
def create_2d_features(signal, delay=10):
    return np.vstack([signal[delay:], signal[:-delay]]).T

X_2d = create_2d_features(y)  # 二维特征空间

# 预训练Isolation Forest模型
iso_forest = IsolationForest(n_estimators=150, contamination=0.05, random_state=42)
iso_forest.fit(X_2d)
anomaly_pred = iso_forest.predict(X_2d)

# 创建网格用于决策边界
xx, yy = np.meshgrid(np.linspace(-10, 10, 200), np.linspace(-10, 10, 200))
Z = iso_forest.decision_function(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

# 初始化画布
fig = plt.figure(figsize=(18, 8))
ax1 = fig.add_subplot(121)  # 时间序列面板
ax2 = fig.add_subplot(122)  # 特征空间面板

# 时间序列面板设置
ax1.set_xlim(0, 10)
ax1.set_ylim(-8, 8)
ax1.set_title('实时信号生成与异常检测', fontsize=12)
ax1.grid(True, linestyle='--', alpha=0.5)

# 特征空间面板设置
ax2.set_xlim(-10, 10)
ax2.set_ylim(-10, 10)
ax2.set_title('Isolation Forest 特征空间分佈', fontsize=12)
ax2.grid(True, linestyle='--', alpha=0.5)

# 绘制决策边界
cmap = ListedColormap(['#FFDDDD', '#DDFFDD', '#DDDDFF'])
ax2.contourf(xx, yy, Z, levels=np.linspace(Z.min(), Z.max(), 10), cmap=cmap, alpha=0.3)
ax2.contour(xx, yy, Z, levels=[0], linewidths=2, colors='darkred')

# 初始化图形元素
# 时间序列面板
ts_line, = ax1.plot([], [], 'blue', alpha=0.5, lw=1)
ts_normal = ax1.scatter([], [], c='green', s=30, edgecolor='k', label='正常')
ts_anomaly = ax1.scatter([], [], c='red', s=60, marker='X', label='异常')

# 特征空间面板
fs_normal = ax2.scatter([], [], c='green', s=30, edgecolor='k', label='正常')
fs_anomaly = ax2.scatter([], [], c='red', s=60, marker='X', label='异常')
fs_trace = ax2.plot([], [], 'gray', alpha=0.2, lw=0.5)[0]

def update(frame):
    # 更新时间序列面板
    current_t = t[:frame+1]
    current_y = y[:frame+1]
    
    ts_line.set_data(current_t, current_y)
    mask = anomaly_pred[:frame+1] == 1
    
    ts_normal.set_offsets(np.c_[current_t[mask], current_y[mask]])
    ts_anomaly.set_offsets(np.c_[current_t[~mask], current_y[~mask]])
    
    ax1.set_xlim(max(0, current_t[-1]-8), min(10, current_t[-1]+2))
    
    # 更新特征空间面板
    if frame >= 10:  # 延迟10帧开始绘制二维特征
        current_features = X_2d[:frame-9]
        current_pred = anomaly_pred[:frame-9]
        
        fs_normal.set_offsets(current_features[current_pred == 1])
        fs_anomaly.set_offsets(current_features[current_pred == -1])
        
        # 绘制特征移动轨迹
        if frame % 20 == 0 and frame > 50:
            fs_trace.set_data(X_2d[frame-50:frame-9, 0], X_2d[frame-50:frame-9, 1])
    
    return ts_line, ts_normal, ts_anomaly, fs_normal, fs_anomaly, fs_trace

ani = FuncAnimation(
    fig,
    update,
    frames=TOTAL_POINTS,
    interval=ANIMATION_SPEED,
    blit=True
)

plt.tight_layout()
plt.show()