import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.display
import os

TARGET_SR = 16000    # 16 kHz matches target ESP32 + INMP441 sampling
N_FFT = 512          # Window size for FFT (~32 ms at 16 kHz)
HOP_LENGTH = 256     # Stride between frames (~16 ms / 50% overlap)
N_MELS = 10          # Mel filter banks
N_MFCC = 13          # TinyML compact fingerprint coefficients

def loadAudio(audio_path, target_sr=16000, duration_s=1.0, normalize=True):
    if not os.path.exists(audio_path):
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    y, sr = librosa.load(audio_path, sr=target_sr, mono=True)

    max_samples = int(target_sr * duration_s)
    if len(y) > max_samples:
        y = y[:max_samples]
    elif len(y) < max_samples:
        y = np.pad(y, (0, max_samples - len(y)), mode='constant')

    if normalize:
        peak = np.max(np.abs(y))
        if peak > 0:
            y = (y / peak) * 0.95

    return y.astype(np.float32), sr

audio_file = "bearing_annormal.wav"  
y, sr = loadAudio(audio_file)

# Step 2: Linear FFT Spectrogram (STFT)
D = librosa.stft(y, n_fft=N_FFT, hop_length=HOP_LENGTH)
linear_spec_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)

# Step 3: Mel Spectrogram (Triangular Filter Banks)
mel_spec = librosa.feature.melspectrogram(
    y=y, sr=sr, n_fft=N_FFT, hop_length=HOP_LENGTH, n_mels=N_MELS
)
mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)

# Step 4: MFCC Fingerprint (DCT Compression)
mfccs = librosa.feature.mfcc(
    y=y, sr=sr, n_fft=N_FFT, hop_length=HOP_LENGTH, n_mfcc=N_MFCC
)

# Fix 1: Compute time array for frames
times = librosa.times_like(mfccs, sr=sr, hop_length=HOP_LENGTH)

# Step 5: Plotting the 5 Stages in one figure
fig, axes = plt.subplots(5, 1, figsize=(10, 15), constrained_layout=True)

# 1. Raw Waveform
librosa.display.waveshow(y, sr=sr, ax=axes[0], color='#1f77b4')
axes[0].set_title("1. Raw Time-Domain Audio (16 kHz PCM)", fontsize=11, fontweight='bold')
axes[0].set_ylabel("Amplitude")
axes[0].set_xlabel("Time (s)")

# 2. Linear Spectrogram (FFT)
img_stft = librosa.display.specshow(
    linear_spec_db, sr=sr, hop_length=HOP_LENGTH, x_axis='time', y_axis='linear',
    ax=axes[1], cmap='magma'
)
axes[1].set_title("2. Linear FFT Spectrogram (Ingredients across all frequencies)", fontsize=11, fontweight='bold')
axes[1].set_ylabel("Freq (Hz)")
fig.colorbar(img_stft, ax=axes[1], format="%+2.0f dB")

# 3. Mel Spectrogram
img_mel = librosa.display.specshow(
    mel_spec_db, sr=sr, hop_length=HOP_LENGTH, x_axis='time', y_axis='mel',
    ax=axes[2], cmap='viridis'
)
axes[2].set_title(f"3. Mel Spectrogram ({N_MELS} Non-Linear Filter Banks)", fontsize=11, fontweight='bold')
axes[2].set_ylabel("Mel Freq")
fig.colorbar(img_mel, ax=axes[2], format="%+2.0f dB")

# 4. MFCC Fingerprint Heatmap
img_mfcc = librosa.display.specshow(
    mfccs, sr=sr, hop_length=HOP_LENGTH, x_axis='time',
    ax=axes[3], cmap='coolwarm'
)
axes[3].set_title(f"4. MFCC 2D Heatmap ({N_MFCC} Compressed Coefficients for TinyML)", fontsize=11, fontweight='bold')
axes[3].set_ylabel("MFCC Index")
fig.colorbar(img_mfcc, ax=axes[3])

# 5. Instantaneous Spectral Envelope Curves (First 4 Time Frames) plotted on axes[4]
num_frames_to_plot = 4
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
markers = ['o', 's', '^', 'D']
coeff_indices = np.arange(mfccs.shape[0])  # 0 to 12

for idx in range(num_frames_to_plot):
    frame_coeffs = mfccs[:, idx]
    timestamp = times[idx]
    
    axes[4].plot(
        coeff_indices, 
        frame_coeffs, 
        label=f"Frame {idx} (t = {timestamp:.3f} s)", 
        color=colors[idx], 
        marker=markers[idx],
        markersize=5,
        linewidth=1.8,
        alpha=0.85
    )

axes[4].set_title(f"5. Instantaneous Spectral Envelope Curves (First {num_frames_to_plot} Frames)", fontsize=11, fontweight='bold')
axes[4].set_xlabel("MFCC Coefficient Index (0 = Energy, 1 = Tilt, 2+ = Structural Modes)", fontsize=10)
axes[4].set_ylabel("Coeff Value", fontsize=10)
axes[4].set_xticks(coeff_indices)
axes[4].grid(True, linestyle=":", alpha=0.6)
axes[4].legend(loc="upper right", fontsize=9)

plt.show()