import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.display
from scipy.fftpack import idct
import os
import glob
import re

st.set_page_config(
    page_title="MIMII Acoustic Inspector",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    section[data-testid="stSidebar"] > div:first-child {
        padding-top: 1rem !important;
        padding-bottom: 1rem !important;
    }
    section[data-testid="stSidebar"] .stSelectbox {
        margin-bottom: -10px;
    }
    div[data-testid="stHorizontalBlock"] button[kind="secondary"] {
        padding: 2px 6px !important;
        height: auto !important;
        min-height: 28px !important;
    }
</style>
""", unsafe_allow_html=True)

AUDIO_ROOT = "bearing/bearingAudioBoosted"
TARGET_SR = 16000
AUDIO_LEN = 1.0
N_FFT = 2048
HOP_LENGTH = 1024
N_MELS = 10
N_MFCC = 13

@st.cache_data
def scan_dataset(root_dir):
    files = glob.glob(os.path.join(root_dir, "**", "*.wav"), recursive=True)
    catalog = []
    pattern = re.compile(
        r"(section_\d+)_([a-zA-Z]+)_(train|test)_(normal|anomaly)_(\d+)(?:_(.+))?\.wav$"
    )

    for f in files:
        fname = os.path.basename(f)
        match = pattern.search(fname)
        if match:
            catalog.append({
                "path": f,
                "filename": fname,
                "section": match.group(1),
                "domain": match.group(2),
                "split": match.group(3),
                "label": match.group(4),
                "idx": match.group(5),
                "attribute": match.group(6) if match.group(6) else "none"
            })
    return catalog

with st.sidebar:
    st.markdown("### Dataset Source")
    audio_root = st.text_input("Source Directory", value=AUDIO_ROOT)

catalog = scan_dataset(audio_root)

if not catalog:
    st.error(f"No matching .wav files found in directory: `{audio_root}`.")
    st.stop()

if "workspace_files" not in st.session_state:
    st.session_state.workspace_files = []
if "minimized_files" not in st.session_state:
    st.session_state.minimized_files = []

@st.cache_data(max_entries=16, show_spinner=False)
def compute_pipeline(file_path, sr_val, len_val, n_fft_val, hop_val, n_mels_val, n_mfcc_val):
    y, _ = librosa.load(file_path, sr=sr_val, mono=True, dtype=np.float32)
    max_len = int(sr_val * len_val)
    if len(y) > max_len:
        y = y[:max_len]
    else:
        y = np.pad(y, (0, max_len - len(y)), mode='constant')

    peak = np.max(np.abs(y))
    if peak > 0:
        y = (y / peak) * 0.95

    D = librosa.stft(y, n_fft=n_fft_val, hop_length=hop_val)
    linear_db = librosa.amplitude_to_db(np.abs(D), ref=np.max).astype(np.float32)

    mel = librosa.feature.melspectrogram(
        y=y, sr=sr_val, n_fft=n_fft_val, hop_length=hop_val, n_mels=n_mels_val
    )
    mel_db = librosa.power_to_db(mel, ref=np.max).astype(np.float32)

    mfcc = librosa.feature.mfcc(
        y=y, sr=sr_val, n_fft=n_fft_val, hop_length=hop_val, n_mfcc=n_mfcc_val
    ).astype(np.float32)
    times = librosa.times_like(mfcc, sr=sr_val, hop_length=hop_val).astype(np.float32)

    return y, linear_db, mel_db, mfcc, times

with st.sidebar:
    st.markdown("### Sample Query Builder")
    st.caption("Construct file paths via official MIMII tokens")

    sections = sorted(list(set(c["section"] for c in catalog)))
    sel_section = st.selectbox("Section", sections) if sections else None

    domains = sorted(list(set(c["domain"] for c in catalog if c["section"] == sel_section)))
    sel_domain = st.selectbox("Domain", domains) if domains else None

    splits = sorted(list(set(c["split"] for c in catalog if c["section"] == sel_section and c["domain"] == sel_domain)))
    sel_split = st.selectbox("Split", splits) if splits else None

    labels = sorted(list(set(c["label"] for c in catalog if c["section"] == sel_section and c["domain"] == sel_domain and c["split"] == sel_split)))
    sel_label = st.selectbox("Health Label", labels) if labels else None

    attributes = sorted(list(set(c["attribute"] for c in catalog if c["section"] == sel_section and c["domain"] == sel_domain and c["split"] == sel_split and c["label"] == sel_label)))
    sel_attribute = st.selectbox("Attribute", attributes) if attributes else None

    matched = [c for c in catalog if (
        c["section"] == sel_section and
        c["domain"] == sel_domain and
        c["split"] == sel_split and
        c["label"] == sel_label and
        c["attribute"] == sel_attribute
    )]
    
    indices = sorted([c["idx"] for c in matched])
    sel_idx = st.selectbox("Recording Index", indices) if indices else None

    target_file = next((c for c in matched if c["idx"] == sel_idx), None)

    if target_file:
        st.code(target_file["filename"], language="text")
        if st.button("Open in Workspace", use_container_width=True):
            if target_file["path"] not in st.session_state.workspace_files and target_file["path"] not in st.session_state.minimized_files:
                st.session_state.workspace_files.append(target_file["path"])
                st.rerun()

    st.markdown("---")

    st.markdown(f"#### Stashed Samples ({len(st.session_state.minimized_files)})")
    for fpath in list(st.session_state.minimized_files):
        fname = os.path.basename(fpath)
        col_m1, col_m2 = st.columns([0.8, 0.2])
        col_m1.caption(f"`{fname}`")
        if col_m2.button("↗", key=f"restore_{fpath}", help="Restore to workspace"):
            st.session_state.minimized_files.remove(fpath)
            st.session_state.workspace_files.append(fpath)
            st.rerun()

head_left, head_right = st.columns([0.8, 0.2])
with head_left:
    st.title("Acoustic Diagnostic Staging Area")
with head_right:
    st.write("")
    if st.session_state.workspace_files:
        if st.button("Clear All", use_container_width=True):
            st.session_state.workspace_files.clear()
            st.rerun()

if not st.session_state.workspace_files:
    st.info("Your workspace is empty. Use the left sidebar to construct and open MIMII samples.")
else:
    num_active = len(st.session_state.workspace_files)
    cols = st.columns(min(num_active, 4))

    for i, fpath in enumerate(st.session_state.workspace_files):
        target_col = cols[i % min(num_active, 4)]
        fname = os.path.basename(fpath)
        is_anomaly = "anomaly" in fname.lower()

        with target_col:
            with st.container(border=True):
                head_col, btn_min, btn_close = st.columns([0.82, 0.09, 0.09])
                with head_col:
                    label_color = "red" if is_anomaly else "green"
                    st.markdown(f"**:{label_color}[{'ANOMALY' if is_anomaly else 'NORMAL'}]** `{fname}`")
                with btn_min:
                    if st.button("🗕", key=f"min_{fpath}", help="Minimize to left tray"):
                        st.session_state.workspace_files.remove(fpath)
                        st.session_state.minimized_files.append(fpath)
                        st.rerun()
                with btn_close:
                    if st.button("✕", key=f"del_{fpath}", help="Close tab"):
                        st.session_state.workspace_files.remove(fpath)
                        st.rerun()

                st.audio(fpath)

                y, linear_db, mel_db, mfccs, times = compute_pipeline(
                    fpath, TARGET_SR, AUDIO_LEN, N_FFT, HOP_LENGTH, N_MELS, N_MFCC
                )

                actual_n_fft = (linear_db.shape[0] - 1) * 2
                actual_n_mels = mel_db.shape[0]
                fft_freqs = librosa.fft_frequencies(sr=TARGET_SR, n_fft=actual_n_fft)
                mel_freqs = librosa.mel_frequencies(n_mels=actual_n_mels, fmin=0.0, fmax=TARGET_SR / 2.0)

                fig, axes = plt.subplots(5, 1, figsize=(6, 13), constrained_layout=True)

                librosa.display.waveshow(
                    y, sr=TARGET_SR, ax=axes[0], 
                    color='#d62728' if is_anomaly else '#1f77b4'
                )
                axes[0].set_title(f"1. Raw Audio ({TARGET_SR // 1000} kHz PCM - {AUDIO_LEN:.1f}s)", fontsize=9, fontweight='bold')
                axes[0].set_xlabel(f"Time (0 to {AUDIO_LEN:.1f}s)", fontsize=8)
                axes[0].set_ylabel("Amplitude", fontsize=8)
                axes[0].set_xlim(0, AUDIO_LEN)
                axes[0].set_ylim(-1.0, 1.0)

                librosa.display.specshow(
                    linear_db, sr=TARGET_SR, hop_length=HOP_LENGTH, 
                    x_axis='time', y_axis='linear', ax=axes[1], cmap='magma'
                )
                axes[1].set_title(f"2. Linear FFT Spectrogram (N_FFT={actual_n_fft})", fontsize=9, fontweight='bold')
                axes[1].set_xlabel("Time (s)", fontsize=8)
                axes[1].set_ylabel(f"Frequency (0 to {TARGET_SR // 2} Hz)", fontsize=8)
                axes[1].set_xlim(0, AUDIO_LEN)

                librosa.display.specshow(
                    mel_db, sr=TARGET_SR, hop_length=HOP_LENGTH, 
                    x_axis='time', y_axis='mel', ax=axes[2], cmap='viridis'
                )
                axes[2].set_title(f"3. Mel Spectrogram ({actual_n_mels} Filter Banks)", fontsize=9, fontweight='bold')
                axes[2].set_xlabel("Time (s)", fontsize=8)
                axes[2].set_ylabel("Mel Frequency", fontsize=8)
                axes[2].set_xlim(0, AUDIO_LEN)

                librosa.display.specshow(
                    mfccs, sr=TARGET_SR, hop_length=HOP_LENGTH, 
                    x_axis='time', ax=axes[3], cmap='coolwarm'
                )
                axes[3].set_title(f"4. MFCC Heatmap (0 to {mfccs.shape[0] - 1} Coeffs, {AUDIO_LEN:.1f}s)", fontsize=9, fontweight='bold')
                axes[3].set_xlabel("Time (s)", fontsize=8)
                axes[3].set_ylabel("MFCC Index", fontsize=8)
                axes[3].set_yticks(range(0, mfccs.shape[0], max(1, mfccs.shape[0] // 6)))
                axes[3].set_xlim(0, AUDIO_LEN)

                num_frames = mfccs.shape[1]
                mid_frame = num_frames // 2
                mid_time = times[mid_frame]
                fft_slice_db = linear_db[:, mid_frame]

                axes[4].plot(
                    fft_freqs, fft_slice_db, 
                    color='lightgray', alpha=0.6, lw=0.8, 
                    label=f"Raw FFT (t={mid_time:.2f}s)"
                )

                slice_configs = [
                    (0.20, ":", 0.6, 1.4),
                    (0.50, "-", 0.95, 2.2),
                    (0.80, "--", 0.6, 1.4)
                ]
                base_color = '#d62728' if is_anomaly else '#1f77b4'

                evaluated_timestamps = []
                for frac, ls, alpha_val, lw_val in slice_configs:
                    frame_idx = min(int(num_frames * frac), num_frames - 1)
                    t_sec = times[frame_idx]
                    evaluated_timestamps.append(f"{t_sec:.2f}s")
                    
                    recon_env = idct(mfccs[:, frame_idx], type=3, n=actual_n_mels, norm='ortho')
                    axes[4].plot(
                        mel_freqs, recon_env,
                        color=base_color, linestyle=ls, alpha=alpha_val, lw=lw_val,
                        label=f"Envelope (t={t_sec:.2f}s)"
                    )

                timestamps_str = ", ".join(evaluated_timestamps)
                axes[4].set_title(
                    f"5. Spectral Envelopes (t = {timestamps_str} | {actual_n_mels} Mel Bands)", 
                    fontsize=9, fontweight='bold'
                )
                axes[4].set_xlabel(f"Frequency (0 to {TARGET_SR // 2} Hz)", fontsize=8)
                axes[4].set_ylabel("Log Energy (dB)", fontsize=8)
                axes[4].set_xlim(0, TARGET_SR // 2)
                axes[4].grid(True, linestyle=":", alpha=0.5)
                axes[4].legend(loc="upper right", fontsize=6.5)

                st.pyplot(fig)
                plt.close(fig)