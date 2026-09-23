'''
from scipy.io import wavfile
import numpy as np

sr, data = wavfile.read("section_00_source_test_anomaly_0027_vel_22.wav")
signal = data.astype(np.float32)

peak = np.max(np.abs(signal))
if peak > 0:
    signal = (signal / peak) * 0.95

amplified_pcm = (signal * 32767).astype(np.int16)
wavfile.write("bearing_annormal.wav", sr, amplified_pcm)
print("Saved boosted file as: audible_bearing_normal.wav")

'''

import os
from pathlib import Path
from scipy.io import wavfile
import numpy as np

SOURCE_FOLDER = "bearing/test"
NEW_OUTPUT_FOLDER = "bearing/bearingAudioBoosted"

TARGET_PEAK = 0.95

def process_and_save_amplified_audio(src_dir, dest_dir, target_peak=0.95):
    src = Path(src_dir)
    dest = Path(dest_dir)

    dest.mkdir(parents=True, exist_ok=True)

    wav_files = list(src.rglob("*.wav"))
    
    if not wav_files:
        print(f"No .wav files found in '{src_dir}'! Check your folder path.")
        return

    total = len(wav_files)
    print(f"Found {total} files. Saving amplified copies to '{dest_dir}'...\n")

    for i, file_path in enumerate(wav_files, 1):
        relative_path = file_path.relative_to(src)
        out_path = dest / relative_path
        out_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            sr, data = wavfile.read(file_path)

            if len(data.shape) > 1:
                signal = data[:, 0].astype(np.float32)
            else:
                signal = data.astype(np.float32)

            max_val = np.max(np.abs(signal))
            if max_val > 0:
                signal = (signal / max_val) * (32767.0 * target_peak)

            boosted_pcm = signal.astype(np.int16)
            wavfile.write(out_path, sr, boosted_pcm)

            if i % 50 == 0 or i == total:
                print(f"[{i}/{total}] Processed and saved: {relative_path}")

        except Exception as e:
            print(f"Skipped {file_path.name} due to error: {e}")

    print(f"\nFinished! All high-volume files are ready in: {dest.resolve()}")

if __name__ == "__main__":
    process_and_save_amplified_audio(SOURCE_FOLDER, NEW_OUTPUT_FOLDER, TARGET_PEAK)