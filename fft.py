import numpy as np
import matplotlib.pyplot as plt
import os
import librosa

# Load audio file
BASE_DIR = os.path.dirname(os.path.abspath(r"C:\Users\goaic\Desktop\plot_fft\audio\machine_sound_100.wav"))
impact, sr = librosa.load(os.path.join(BASE_DIR, "machine_sound_100.wav"), sr=None)

# --- TIME DOMAIN PLOT (Waveform) ---
plt.figure(figsize=(12, 8))

# Subplot 1: Time-domain waveform
plt.subplot(2, 1, 1)
time = np.arange(len(impact)) / sr
plt.plot(time, impact, color='b')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.title('Time Domain: Impact Sound Waveform')
plt.grid()

# --- FREQUENCY DOMAIN PLOT (FFT) ---
plt.subplot(2, 1, 2)
ft = np.fft.rfft(impact)
magnitude = np.abs(ft)
frequency = np.fft.rfftfreq(len(impact), d=1/sr)
plt.plot(frequency, magnitude, color='r')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Magnitude')
plt.title('Frequency Domain: FFT of Impact Sound')
plt.grid()

plt.tight_layout()
plt.show()