import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from sklearn.ensemble import IsolationForest
from matplotlib.colors import ListedColormap
import librosa
import os

# Load your machine audio file
audio_path = os.path.abspath(r"C:\Users\goaic\Desktop\plot_fft\audio\machine_anomaly.wav")

try:
    # Load audio file
    y_audio, sr = librosa.load(audio_path, sr=None)
    print(f"Audio loaded successfully!")
    print(f"Sample rate: {sr} Hz")
    print(f"Duration: {len(y_audio)/sr:.2f} seconds")
    print(f"Total samples: {len(y_audio)}")
    
    # Optional: Downsample for faster processing
    downsample_factor = max(1, len(y_audio) // 2000)  # Limit to ~2000 points for animation
    y_audio_downsampled = y_audio[::downsample_factor]
    
    # Use real audio data
    TOTAL_POINTS = len(y_audio_downsampled)
    t = np.linspace(0, len(y_audio)/sr, TOTAL_POINTS)
    y = y_audio_downsampled
    
    # Normalize audio data
    y = (y - np.mean(y)) / np.std(y)  # Standardize
    
    USE_REAL_AUDIO = True
    
except FileNotFoundError:
    print(f"Audio file not found at: {audio_path}")
    print("Using synthetic data instead...")
    USE_REAL_AUDIO = False
except Exception as e:
    print(f"Error loading audio: {e}")
    print("Using synthetic data instead...")
    USE_REAL_AUDIO = False

# Animation speed - adjust based on your preference
ANIMATION_SPEED = 50  # milliseconds between frames

# Create 2D features (time delay embedding)
def create_2d_features(signal, delay=10):
    return np.vstack([signal[delay:], signal[:-delay]]).T

X_2d = create_2d_features(y)

# Train Isolation Forest model
iso_forest = IsolationForest(n_estimators=150, contamination=0.03, random_state=42)
iso_forest.fit(X_2d)
anomaly_pred = iso_forest.predict(X_2d)

# Create grid for decision boundary
y_min, y_max = np.min(y), np.max(y)
margin = (y_max - y_min) * 0.2
xx, yy = np.meshgrid(
    np.linspace(y_min - margin, y_max + margin, 200), 
    np.linspace(y_min - margin, y_max + margin, 200)
)
Z = iso_forest.decision_function(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

# Initialize canvas with wider figure and adjusted spacing
fig = plt.figure(figsize=(20, 8))  # Made figure wider
ax1 = fig.add_subplot(121)  # Time series panel
ax2 = fig.add_subplot(122)  # Feature space panel

# Add more space between subplots
plt.subplots_adjust(wspace=0.4, left=0.08, right=0.95)  # Better spacing and margins

# Time series panel setup
ax1.set_xlim(0, t[-1])
ax1.set_ylim(y_min - margin, y_max + margin)
title = 'Real-time Machine Audio Anomaly Detection'
ax1.set_title(title, fontsize=12)
ax1.set_xlabel('Time (seconds)')
ax1.set_ylabel('Amplitude')
ax1.grid(True, linestyle='--', alpha=0.5)

# Feature space panel setup
ax2.set_xlim(y_min - margin, y_max + margin)
ax2.set_ylim(y_min - margin, y_max + margin)
ax2.set_title('Isolation Forest Feature Space Distribution', fontsize=12)
ax2.set_xlabel('X(t)')
ax2.set_ylabel('X(t-τ)')
ax2.grid(True, linestyle='--', alpha=0.5)

# Draw decision boundary
cmap = ListedColormap(['#FFDDDD', '#DDFFDD', '#DDDDFF'])
ax2.contourf(xx, yy, Z, levels=np.linspace(Z.min(), Z.max(), 10), cmap=cmap, alpha=0.3)
ax2.contour(xx, yy, Z, levels=[0], linewidths=2, colors='darkred')

# Initialize graphic elements
# Time series panel
ts_line, = ax1.plot([], [], 'blue', alpha=0.7, lw=1)
ts_normal = ax1.scatter([], [], c='green', s=30, edgecolor='k', label='Normal', alpha=0.8)
ts_anomaly = ax1.scatter([], [], c='red', s=60, marker='X', label='Anomaly', alpha=0.9)

# Feature space panel
fs_normal = ax2.scatter([], [], c='green', s=30, edgecolor='k', label='Normal', alpha=0.8)
fs_anomaly = ax2.scatter([], [], c='red', s=60, marker='X', label='Anomaly', alpha=0.9)
fs_trace = ax2.plot([], [], 'gray', alpha=0.3, lw=0.8)[0]

# Add legends
ax1.legend(loc='upper right')
ax2.legend(loc='upper right')

def update(frame):
    # Update time series panel
    current_t = t[:frame+1]
    current_y = y[:frame+1]
    
    ts_line.set_data(current_t, current_y)
    mask = anomaly_pred[:frame+1] == 1
    
    ts_normal.set_offsets(np.c_[current_t[mask], current_y[mask]])
    ts_anomaly.set_offsets(np.c_[current_t[~mask], current_y[~mask]])
    
    # Dynamic window for time series (show last portion of data)
    window_size = t[-1] * 0.8  # Show 80% of total time range
    if current_t[-1] > window_size:
        ax1.set_xlim(current_t[-1] - window_size, current_t[-1] + window_size * 0.1)
    
    # Update feature space panel
    if frame >= 10:  # Start after delay
        current_features = X_2d[:frame-9]
        current_pred = anomaly_pred[:frame-9]
        
        fs_normal.set_offsets(current_features[current_pred == 1])
        fs_anomaly.set_offsets(current_features[current_pred == -1])
        
        # Draw feature trajectory
        trace_length = min(50, frame - 10)
        if frame > 20:
            trace_start = max(0, frame - 9 - trace_length)
            trace_end = frame - 9
            fs_trace.set_data(
                X_2d[trace_start:trace_end, 0], 
                X_2d[trace_start:trace_end, 1]
            )
    
    return ts_line, ts_normal, ts_anomaly, fs_normal, fs_anomaly, fs_trace

# Create animation
ani = FuncAnimation(
    fig, update, 
    frames=TOTAL_POINTS, 
    interval=ANIMATION_SPEED, 
    blit=True
)


# Add info text at bottom
try:
    info_text = f"Audio: {os.path.basename(audio_path)}"
except:
    info_text = "Audio: Synthetic Data"
fig.suptitle(info_text, fontsize=10, y=0.02)

plt.show()

# Optional: Save animation as GIF
# ani.save('machine_audio_anomaly.gif', writer='pillow', fps=20)