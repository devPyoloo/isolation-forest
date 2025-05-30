import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from sklearn.ensemble import IsolationForest
from matplotlib.colors import ListedColormap
import librosa
import os
import threading
import time
import pygame

# Initialize pygame mixer for audio playback
pygame.mixer.init()

# Load your machine audio file
audio_path = os.path.abspath(r"C:\Users\goaic\Desktop\plot_fft\audio\machine_anomaly.wav")

try:
    # Load audio file
    y_audio, sr = librosa.load(audio_path, sr=None)
    print(f"Audio loaded successfully!")
    print(f"Sample rate: {sr} Hz")
    print(f"Duration: {len(y_audio)/sr:.2f} seconds")
    print(f"Total samples: {len(y_audio)}")
    
    # Calculate audio duration
    audio_duration = len(y_audio) / sr
    
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
    
    # Load audio for playback
    try:
        pygame.mixer.music.load(audio_path)
        AUDIO_AVAILABLE = True
        print("Audio file loaded for playback!")
    except:
        AUDIO_AVAILABLE = False
        print("Could not load audio for playback")
    
except FileNotFoundError:
    print(f"Audio file not found at: {audio_path}")
    print("Using synthetic data instead...")
    USE_REAL_AUDIO = False
    AUDIO_AVAILABLE = False
    
    # Create synthetic data with known duration
    audio_duration = 10.0  # 10 seconds
    TOTAL_POINTS = 2000
    t = np.linspace(0, audio_duration, TOTAL_POINTS)
    y = np.sin(2 * np.pi * 1 * t) + 0.5 * np.sin(2 * np.pi * 3 * t) + 0.1 * np.random.randn(TOTAL_POINTS)
    y = (y - np.mean(y)) / np.std(y)
    
except Exception as e:
    print(f"Error loading audio: {e}")
    print("Using synthetic data instead...")
    USE_REAL_AUDIO = False
    AUDIO_AVAILABLE = False
    
    # Create synthetic data
    audio_duration = 10.0
    TOTAL_POINTS = 2000
    t = np.linspace(0, audio_duration, TOTAL_POINTS)
    y = np.sin(2 * np.pi * 1 * t) + 0.5 * np.sin(2 * np.pi * 3 * t) + 0.1 * np.random.randn(TOTAL_POINTS)
    y = (y - np.mean(y)) / np.std(y)

# Fast animation for real-time feel
ANIMATION_SPEED = 16  # ~60 FPS for smooth animation
print(f"Animation speed: {ANIMATION_SPEED} ms per frame")

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

# Time-based synchronization variables
animation_start_time = None
audio_started = False

def start_audio_sync():
    """Start audio playback synchronized with animation"""
    global animation_start_time
    if AUDIO_AVAILABLE:
        # Wait for animation to actually start
        while animation_start_time is None:
            time.sleep(0.001)
        
        # Start audio immediately when animation starts
        pygame.mixer.music.play()
        print(f"Audio started at: {time.time() - animation_start_time:.3f}s after animation")

def update(frame):
    global audio_started, animation_start_time
    
    # Record actual animation start time on first frame
    if animation_start_time is None:
        animation_start_time = time.time()
        print(f"Animation started at frame {frame}")
    
    # Start audio synchronization on first frame
    if not audio_started and AUDIO_AVAILABLE:
        audio_thread = threading.Thread(target=start_audio_sync)
        audio_thread.daemon = True
        audio_thread.start()
        audio_started = True
    
    # Calculate current time based on actual elapsed time
    current_elapsed_time = time.time() - animation_start_time if animation_start_time else 0
    
    # Find the corresponding data index based on elapsed time
    # Add small offset to account for processing delays
    sync_offset = 0.05  # 50ms offset to account for system delays
    adjusted_time = current_elapsed_time - sync_offset
    
    # Find current index based on time instead of frame
    if adjusted_time <= 0:
        current_index = 0
    else:
        current_index = int((adjusted_time / audio_duration) * TOTAL_POINTS)
        current_index = min(current_index, TOTAL_POINTS - 1)
    
    # Update time series panel based on current time position
    current_t = t[:current_index+1]
    current_y = y[:current_index+1]
    
    if len(current_t) > 0:
        ts_line.set_data(current_t, current_y)
        mask = anomaly_pred[:current_index+1] == 1
        
        if np.any(mask):
            ts_normal.set_offsets(np.c_[current_t[mask], current_y[mask]])
        else:
            ts_normal.set_offsets(np.empty((0, 2)))
            
        if np.any(~mask):
            ts_anomaly.set_offsets(np.c_[current_t[~mask], current_y[~mask]])
        else:
            ts_anomaly.set_offsets(np.empty((0, 2)))
        
        # Dynamic window for time series (show last portion of data)
        window_size = t[-1] * 0.8  # Show 80% of total time range
        if current_t[-1] > window_size:
            ax1.set_xlim(current_t[-1] - window_size, current_t[-1] + window_size * 0.1)
    
    # Update feature space panel
    if current_index >= 10:  # Start after delay
        feature_index = current_index - 9
        current_features = X_2d[:feature_index]
        current_pred = anomaly_pred[:feature_index]
        
        if len(current_features) > 0:
            normal_mask = current_pred == 1
            anomaly_mask = current_pred == -1
            
            if np.any(normal_mask):
                fs_normal.set_offsets(current_features[normal_mask])
            else:
                fs_normal.set_offsets(np.empty((0, 2)))
                
            if np.any(anomaly_mask):
                fs_anomaly.set_offsets(current_features[anomaly_mask])
            else:
                fs_anomaly.set_offsets(np.empty((0, 2)))
            
            # Draw feature trajectory
            trace_length = min(50, feature_index)
            if feature_index > 20:
                trace_start = max(0, feature_index - trace_length)
                trace_end = feature_index
                fs_trace.set_data(
                    X_2d[trace_start:trace_end, 0], 
                    X_2d[trace_start:trace_end, 1]
                )
    
    # Display sync information
    audio_time = current_elapsed_time
    data_time = (current_index / TOTAL_POINTS) * audio_duration if TOTAL_POINTS > 0 else 0
    sync_diff = abs(audio_time - data_time)
    
    ax1.set_title(f'{title} | Time: {data_time:.2f}s | Sync Δ: {sync_diff:.3f}s', fontsize=12)
    
    return ts_line, ts_normal, ts_anomaly, fs_normal, fs_anomaly, fs_trace

# Create animation with fast refresh rate
ani = FuncAnimation(
    fig, update, 
    frames=range(10000),  # Large number to keep animation running
    interval=ANIMATION_SPEED, 
    blit=True,
    repeat=False,
    cache_frame_data=False  # Don't cache frames for real-time performance
)

# Add cleanup function for audio
def on_close(event):
    if AUDIO_AVAILABLE:
        pygame.mixer.music.stop()
        pygame.mixer.quit()
    plt.close('all')

fig.canvas.mpl_connect('close_event', on_close)

# Add info text at bottom
try:
    info_text = f"Audio: {os.path.basename(audio_path)} | Sync: TIME-BASED | Duration: {audio_duration:.2f}s"
except:
    info_text = f"Audio: Synthetic Data | Sync: TIME-BASED | Duration: {audio_duration:.2f}s"

fig.suptitle(info_text, fontsize=10, y=0.02)

print(f"\nTime-Based Synchronization Info:")
print(f"- Audio duration: {audio_duration:.2f} seconds")
print(f"- Total data points: {TOTAL_POINTS}")
print(f"- Animation refresh: {ANIMATION_SPEED} ms (~{1000//ANIMATION_SPEED} FPS)")
print(f"- Sync method: Real-time based on elapsed time")
print(f"- Sync offset: 50ms compensation for system delays")

plt.show()