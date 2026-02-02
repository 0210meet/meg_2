#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
run_thal_cortex_spike_ccg_strong.py

Paper-style bidirectional spike–spike CCG analysis
with STRONG morphology-constrained IED detection.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks, peak_widths
from scipy.stats import zscore
from mne.filter import filter_data

# ===================== User configuration =====================

SAVE_DIR = "/data/shared_home/tlm/Project/MEG-C/results2"
os.makedirs(SAVE_DIR, exist_ok=True)

IED_BAND = (20, 80)

Z_THRESH = 3.5
MIN_INTERVAL_MS = 20

RISE_MAX_MS = 20
FWHM_MIN_MS = 20
FWHM_MAX_MS = 70
SLOW_WAVE_RATIO = 0.5

WINDOW_MS = 50
BIN_MS = 2
N_SHUFFLE = 1000


# ===================== Preprocessing =====================

def preprocess_signal(sig, sfreq):
    """对信号进行带通滤波和Z-score标准化"""
    sig_filt = filter_data(
        sig, sfreq,
        IED_BAND[0], IED_BAND[1],
        method="fir",
        phase="zero-double"
    )
    return sig_filt, zscore(sig_filt)


def leakage_reduction_regression(signal_A, signal_B, sfreq):
    """泄漏减少回归：从信号A中去除信号B的共线部分"""
    proj_A_on_B = np.dot(signal_A, signal_B) / np.dot(signal_B, signal_B) * signal_B
    proj_B_on_A = np.dot(signal_B, signal_A) / np.dot(signal_A, signal_A) * signal_A

    signal_A_clean = signal_A - proj_A_on_B
    signal_B_clean = signal_B - proj_B_on_A

    return signal_A_clean, signal_B_clean


# ===================== Morphology-based IED detector =====================

def detect_ied_morphology(sig_raw, sig_filt_z, sfreq):
    env = np.abs(sig_filt_z)
    min_dist = int(MIN_INTERVAL_MS * sfreq / 1000)

    cand_peaks, _ = find_peaks(env, height=Z_THRESH, distance=min_dist)
    if len(cand_peaks) == 0:
        return np.array([], dtype=int)

    raw_amp_thr = np.percentile(np.abs(sig_raw), 95)
    keep = []

    for cp in cand_peaks:

        w = int(0.05 * sfreq)
        if cp - w < 0 or cp + w >= len(sig_raw):
            continue

        local_raw = sig_raw[cp - w:cp + w]
        if np.ptp(local_raw) < raw_amp_thr:
            continue

        peak_idx = np.argmax(local_raw)
        p20, p80 = np.percentile(local_raw, [20, 80])

        try:
            i20 = np.where(local_raw[:peak_idx] <= p20)[0][-1]
            i80 = np.where(local_raw[:peak_idx] <= p80)[0][-1]
        except IndexError:
            continue

        rise_ms = (i80 - i20) / sfreq * 1000
        if rise_ms > RISE_MAX_MS:
            continue

        widths, _, _, _ = peak_widths(local_raw, [peak_idx], rel_height=0.5)
        fwhm_ms = widths[0] / sfreq * 1000
        if not (FWHM_MIN_MS <= fwhm_ms <= FWHM_MAX_MS):
            continue

        slow_len = int(0.25 * sfreq)
        if cp + slow_len >= len(sig_raw):
            continue

        slow_raw = sig_raw[cp:cp + slow_len]
        slow_power = np.sum(np.abs(slow_raw))
        fast_power = np.sum(env[cp:cp + slow_len])
        if slow_power < SLOW_WAVE_RATIO * fast_power:
            continue

        keep.append(cp)

    return np.array(keep, dtype=int)


# ===================== CCG utilities =====================

def compute_ccg(ref_times, target_times):
    window = WINDOW_MS / 1000
    bin_sec = BIN_MS / 1000
    bins = np.arange(-window, window + bin_sec, bin_sec)

    diffs = []
    for t in ref_times:
        diffs.extend(target_times - t)

    diffs = np.asarray(diffs)
    diffs = diffs[np.abs(diffs) <= window]

    hist, edges = np.histogram(diffs, bins=bins)
    centers = (edges[:-1] + edges[1:]) / 2
    return centers, hist


def shuffle_spikes(times, tmax):
    return np.random.uniform(0, tmax, size=len(times))


def ccg_with_shuffle(ref_times, target_times, tmax):
    lags, ccg = compute_ccg(ref_times, target_times)
    null = []

    for _ in range(N_SHUFFLE):
        ref_shuf = shuffle_spikes(ref_times, tmax)
        _, h = compute_ccg(ref_shuf, target_times)
        null.append(h)

    null = np.asarray(null)
    low = np.percentile(null, 2.5, axis=0)
    high = np.percentile(null, 97.5, axis=0)

    return lags, ccg, low, high


# ===================== Plotting =====================

def plot_bidirectional_ccg(res, tag):
    lags = res["CT"]["lags"] * 1000
    ccg_ct = res["CT"]["ccg"]
    low_ct = res["CT"]["low"]
    high_ct = res["CT"]["high"]

    ccg_tc = res["TC"]["ccg"]
    low_tc = res["TC"]["low"]
    high_tc = res["TC"]["high"]

    di = res["DI"]

    fig, axes = plt.subplots(2, 1, figsize=(7, 6), sharex=True)

    # Cortex → Thalamus
    axes[0].bar(lags, ccg_ct, width=BIN_MS, color="k")
    axes[0].fill_between(lags, low_ct, high_ct, color="gray", alpha=0.3)
    axes[0].axvline(0, color="r", linestyle="--")
    axes[0].set_title("Cortex → Thalamus")

    # Thalamus → Cortex
    axes[1].bar(lags, ccg_tc, width=BIN_MS, color="k")
    axes[1].fill_between(lags, low_tc, high_tc, color="gray", alpha=0.3)
    axes[1].axvline(0, color="r", linestyle="--")
    axes[1].set_title("Thalamus → Cortex")

    axes[1].set_xlabel("Lag (ms)")
    axes[0].set_ylabel("Count")
    axes[1].set_ylabel("Count")

    fig.suptitle(f"{tag} | Directionality Index = {di:.2f}", fontsize=12)
    plt.tight_layout()
    plt.show()


# ===================== Main pipeline =====================

def run_bidirectional_spike_ccg(cortex_ts, thal_ts, sfreq, tag="run"):
    # 1. 预处理信号并进行 Z-score 标准化
    cortex_filt, cortex_z = preprocess_signal(cortex_ts, sfreq)
    thal_filt, thal_z = preprocess_signal(thal_ts, sfreq)

    # 2. 去泄漏回归
    cortex_clean, thal_clean = leakage_reduction_regression(cortex_filt, thal_filt, sfreq)

    # 3. 使用 morphology 进行 IED 检测
    cortex_spikes = detect_ied_morphology(cortex_ts, cortex_z, sfreq)
    thal_spikes = detect_ied_morphology(thal_ts, thal_z, sfreq)

    print(f"[{tag}] Strong IEDs — Cortex: {len(cortex_spikes)}, Thalamus: {len(thal_spikes)}")

    if len(cortex_spikes) < 5 or len(thal_spikes) < 5:
        print("⚠️ Too few IEDs, skipping CCG")
        return None

    # 4. 计算 CCG
    cortex_times = cortex_spikes / sfreq
    thal_times = thal_spikes / sfreq
    tmax = len(cortex_ts) / sfreq

    res = {}

    lags, ccg, low, high = ccg_with_shuffle(cortex_times, thal_times, tmax)
    res["CT"] = dict(lags=lags, ccg=ccg, low=low, high=high)

    lags2, ccg2, low2, high2 = ccg_with_shuffle(thal_times, cortex_times, tmax)
    res["TC"] = dict(lags=lags2, ccg=ccg2, low=low2, high=high2)

    post = (lags > 0) & (lags <= 0.1)
    ct_post = np.sum((ccg - high)[post] * ((ccg - high)[post] > 0))
    tc_post = np.sum((ccg2 - high2)[post] * ((ccg2 - high2)[post] > 0))

    di = (ct_post - tc_post) / (ct_post + tc_post + 1e-6)
    res["DI"] = di

    plot_bidirectional_ccg(res, tag)
    return res


# ===================== Entry =====================

if __name__ == "__main__":
    print("Import and call run_bidirectional_spike_ccg(...)")
