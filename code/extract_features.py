import os
import glob
import re
import argparse
from pathlib import Path
import numpy as np
import librosa

TARGET_SR = 16000
DURATION_S = 1.0
N_FFT = 512
HOP_LENGTH = 256
N_MELS = 40
N_MFCC = 13

FILENAME_PATTERN = re.compile(
    r"(section_\d+)_([a-zA-Z]+)_(train|test)_(normal|anomaly)_(\d+)(?:_(.+))?\.wav$"
)

def extract_features(file_path, target_sr=TARGET_SR, duration_s=DURATION_S, n_fft=N_FFT, hop_length=HOP_LENGTH, n_mels=N_MELS, n_mfcc=N_MFCC):
    y, sr = librosa.load(file_path, sr=target_sr, mono=True, dtype=np.float32)

    max_samples = int(target_sr * duration_s)
    if len(y) > max_samples:
        y = y[:max_samples]
    else:
        y = np.pad(y, (0, max_samples - len(y)), mode='constant')

    peak = np.max(np.abs(y))
    if peak > 0:
        y = (y / peak) * 0.95

    # 1. Linear STFT
    D = librosa.stft(y, n_fft=n_fft, hop_length=hop_length)
    linear_db = librosa.amplitude_to_db(np.abs(D), ref=np.max).astype(np.float32)

    # 2. Mel Spectrogram
    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels)
    mel_db = librosa.power_to_db(mel, ref=np.max).astype(np.float32)

    # 3. MFCCs
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mfcc=n_mfcc).astype(np.float32)

    return {
        "mfcc": mfcc,
        "mel_db": mel_db,
        "linear_db": linear_db,
        "audio_pcm": y
    }

def process_folder(input_dir, output_dir, duration_s=DURATION_S):
    in_path = Path(input_dir)
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    wav_files = sorted(list(in_path.rglob("*.wav")))
    total_files = len(wav_files)

    if total_files == 0:
        print(f"No .wav files found in '{input_dir}'")
        return

    print(f"Found {total_files} .wav files in '{input_dir}'.")
    print(f"Extracting features from {duration_s}s slices (SR={TARGET_SR}Hz, N_MFCC={N_MFCC}, N_MELS={N_MELS})...\n")

    all_mfccs = []
    all_labels = []
    all_sections = []
    all_filenames = []

    for idx, fpath in enumerate(wav_files, 1):
        fname = fpath.name
        match = FILENAME_PATTERN.search(fname)
        
        section = match.group(1) if match else "unknown"
        label = match.group(4) if match else ("anomaly" if "anomaly" in fname.lower() else "normal")

        try:
            feats = extract_features(str(fpath), duration_s=duration_s)

            # Save individual .npz file
            rel_path = fpath.relative_to(in_path).with_suffix(".npz")
            single_out = out_path / rel_path
            single_out.parent.mkdir(parents=True, exist_ok=True)

            np.savez_compressed(
                single_out,
                mfcc=feats["mfcc"],
                mel_db=feats["mel_db"],
                section=section,
                label=label,
                filename=fname
            )

            all_mfccs.append(feats["mfcc"])
            all_labels.append(1 if label == "anomaly" else 0)
            all_sections.append(section)
            all_filenames.append(fname)

            if idx % 100 == 0 or idx == total_files:
                print(f"[{idx}/{total_files}] Processed: {fname}")

        except Exception as e:
            print(f"Error processing {fname}: {e}")

    # Save consolidated dataset archive for direct training/evaluation
    consolidated_file = out_path / "dataset_features_1s.npz"
    np.savez_compressed(
        consolidated_file,
        mfcc_matrix=np.array(all_mfccs),      # Shape: (N_samples, N_MFCC, N_frames)
        labels=np.array(all_labels),          # 0 = normal, 1 = anomaly
        sections=np.array(all_sections),
        filenames=np.array(all_filenames)
    )

    print(f"\nCompleted! Saved individual .npz files in: {out_path.resolve()}")
    print(f"Consolidated dataset bundle saved at: {consolidated_file.resolve()}")
    print(f"Consolidated matrix shape: {np.array(all_mfccs).shape} (Samples, MFCCs, TimeFrames)")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract and save 1-second audio coefficients to disk")
    parser.add_argument("--input_dir", type=str, default="bearing/bearingAudioBoosted", help="Source folder containing .wav files")
    parser.add_argument("--output_dir", type=str, default="extracted_features", help="Output directory to save coefficients")
    parser.add_argument("--duration", type=float, default=1.0, help="Duration of audio slice in seconds (default: 1.0)")
    
    args = parser.parse_args()
    process_folder(args.input_dir, args.output_dir, args.duration)
