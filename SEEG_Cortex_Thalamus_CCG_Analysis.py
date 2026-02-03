import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks, peak_widths
from scipy.stats import zscore
from mne.filter import filter_data

# ===================== 全局配置 =====================
IED_BAND = (20, 80)
Z_THRESH = 3.5
MIN_INTERVAL_MS = 20

# 形态学检测参数
RISE_MAX_MS = 20
FWHM_MIN_MS = 20
FWHM_MAX_MS = 70
SLOW_WAVE_RATIO = 0.5

# 统计分析参数 (适配 500Hz MEG)
WINDOW_MS = 200  # 观察前后 200ms
BIN_MS = 4  # 500Hz 对应 2ms/sample，10ms Bin 包含 5 个采样点
N_SHUFFLE = 1000

def leakage_reduction_regression(signal_A, signal_B, sfreq):
    """泄漏减少回归：从信号A中去除信号B的共线部分"""
    proj_A_on_B = np.dot(signal_A, signal_B) / np.dot(signal_B, signal_B) * signal_B
    proj_B_on_A = np.dot(signal_B, signal_A) / np.dot(signal_A, signal_A) * signal_A

    signal_A_clean = signal_A - proj_A_on_B
    signal_B_clean = signal_B - proj_B_on_A

    return signal_A_clean, signal_B_clean

# ===================== 核心算法模块 (保持不变) =====================
def preprocess_signal(sig, sfreq):
    sig_filt = filter_data(sig, sfreq, IED_BAND[0], IED_BAND[1], method="fir", phase="zero-double", verbose=False)
    return sig_filt, zscore(sig_filt)


def detect_ied_morphology(sig_raw, sig_filt_z, sfreq):
    env = np.abs(sig_filt_z)
    min_dist = int(MIN_INTERVAL_MS * sfreq / 1000)
    cand_peaks, _ = find_peaks(env, height=Z_THRESH, distance=min_dist)
    if len(cand_peaks) == 0: return np.array([], dtype=int)

    raw_amp_thr = np.percentile(np.abs(sig_raw), 95)
    keep = []
    for cp in cand_peaks:
        w = int(0.05 * sfreq)
        if cp - w < 0 or cp + w >= len(sig_raw): continue
        local_raw = sig_raw[cp - w:cp + w]
        if np.ptp(local_raw) < raw_amp_thr: continue
        peak_idx = np.argmax(local_raw)
        widths, _, _, _ = peak_widths(local_raw, [peak_idx], rel_height=0.5)
        fwhm_ms = widths[0] / sfreq * 1000
        if (FWHM_MIN_MS <= fwhm_ms <= FWHM_MAX_MS): keep.append(cp)
    return np.array(keep, dtype=int)


# ===================== 整合分析模块 (CCG + AER) =====================

def run_spike_propagation_analysis(cortex_ts, thal_ts, sfreq, tag="run", save_dir="."):
    """
    整合形态学检测、Shuffle-CCG 和 AER 提取
    """
    # 1. 检测皮层棘波 (触发源)
    _, cortex_z = preprocess_signal(cortex_ts, sfreq)
    cortex_spikes = detect_ied_morphology(cortex_ts, cortex_z, sfreq)

    # 2. 检测丘脑棘波 (目标源)
    _, thal_z = preprocess_signal(thal_ts, sfreq)
    thal_spikes = detect_ied_morphology(thal_ts, thal_z, sfreq)

    if len(cortex_spikes) < 5 or len(thal_spikes) < 5:
        print(f"⚠️ [{tag}] 棘波数量过少 (C:{len(cortex_spikes)} T:{len(thal_spikes)}), 跳过分析")
        return None

    tmax = len(cortex_ts) / sfreq
    bin_sec = BIN_MS / 1000
    win_sec = WINDOW_MS / 1000
    bins = np.arange(-win_sec, win_sec + bin_sec, bin_sec)
    bin_centers = (bins[:-1] + bins[1:]) / 2

    ref_times = cortex_spikes / sfreq
    target_times = thal_spikes / sfreq

    # --- A. 计算 CCG ---
    diffs = []
    for rt in ref_times:
        matches = target_times[(target_times >= rt - win_sec) & (target_times <= rt + win_sec)]
        diffs.extend(matches - rt)

    real_counts, _ = np.histogram(diffs, bins=bins)
    real_rate = real_counts / len(ref_times) / bin_sec

    # --- B. Shuffle 检验 ---
    null_rates = []
    for _ in range(N_SHUFFLE):
        shuf_ref = np.random.uniform(0, tmax, size=len(ref_times))
        shuf_diffs = []
        for sr in shuf_ref:
            matches = target_times[(target_times >= sr - win_sec) & (target_times <= sr + win_sec)]
            shuf_diffs.extend(matches - sr)
        h, _ = np.histogram(shuf_diffs, bins=bins)
        null_rates.append(h / len(ref_times) / bin_sec)
    upper_95 = np.percentile(null_rates, 97.5, axis=0)

    # --- C. 潜伏期与增强计算 ---
    sig_mask = real_rate > upper_95
    post_0 = bin_centers[sig_mask & (bin_centers > 0)]
    latency = post_0[0] * 1000 if len(post_0) > 0 else None

    # 计算 DI (方向性指数) - 窗口 8ms 到 40ms
    mask_di_post = (bin_centers >= 0.008) & (bin_centers <= 0.040)
    mask_di_pre = (bin_centers <= -0.008) & (bin_centers >= -0.040)

    rate_post_sum = np.sum(real_rate[mask_di_post])
    rate_pre_sum = np.sum(real_rate[mask_di_pre])

    # DI 公式: (右边和 - 左边和) / (右边和 + 左边和)
    # DI > 0 代表皮层主导; DI < 0 代表丘脑主导
    di_value = (rate_post_sum - rate_pre_sum) / (rate_post_sum + rate_pre_sum + 1e-6)

    mask_8_40 = (bin_centers >= 0.008) & (bin_centers <= 0.040)
    base_rate = np.mean(real_rate[bin_centers < 0]) + 1e-6
    enhancement = np.max(real_rate[mask_8_40]) / base_rate if any(mask_8_40) else 0

    # --- D. AER 计算 (原始电压叠加) ---
    epochs = []
    half_win_samples = int(win_sec * sfreq)
    for cp in cortex_spikes:
        if cp - half_win_samples >= 0 and cp + half_win_samples < len(thal_ts):
            segment = thal_ts[cp - half_win_samples: cp + half_win_samples].copy()
            # 基线校正：前20%窗口
            segment -= np.mean(segment[:int(0.2 * len(segment))])
            epochs.append(segment)

    aer_mean = np.mean(epochs, axis=0) if epochs else np.zeros(2 * half_win_samples)
    aer_times = np.linspace(-win_sec, win_sec, len(aer_mean)) * 1000

    results = {
        "centers": bin_centers * 1000, "rate": real_rate, "upper": upper_95,
        "latency": latency, "enhancement": enhancement, "sig_mask": sig_mask,
        "aer_mean": aer_mean, "aer_times": aer_times, "n_spikes": len(ref_times),
        "di": di_value  # 新增返回 DI
    }

    plot_propagation_results(results, tag, save_dir)
    return results


def plot_propagation_results(res, tag, save_dir="."):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    # CCG Plot
    ax1.bar(res['centers'], res['rate'], width=BIN_MS, color='steelblue', alpha=0.6, label='Thalamic Spike Rate')
    ax1.plot(res['centers'], res['upper'], color='gray', linestyle='--', label='95% Shuffle CI')
    ax1.scatter(res['centers'][res['sig_mask']], res['rate'][res['sig_mask']], color='red', s=15, zorder=3)

    # 在图形中标注DI和主导方向
    di_val = res['di']
    direction_str = "Cortex -> Thal" if di_val > 0 else "Thal -> Cortex"
    di_text = f"DI: {di_val:.3f} ({direction_str})"

    # 将 DI 标注在图表右上角
    ax1.text(0.95, 0.9, di_text, transform=ax1.transAxes, fontsize=12,
             fontweight='bold', color='darkred', ha='right', bbox=dict(facecolor='white', alpha=0.5))

    if res['latency']:
        ax1.axvline(res['latency'], color='red', alpha=0.5, linestyle=':')
        ax1.text(res['latency'] + 5, ax1.get_ylim()[1] * 0.7, f"Onset: {res['latency']:.1f}ms", color='red',
                 fontweight='bold')

    ax1.axvline(0, color='black', linewidth=1)
    ax1.set_ylabel('Rate (Hz)')
    ax1.set_title(f"Propagation: Cortex -> Thalamus | {tag}\nEnhancement (8-40ms): {res['enhancement']:.2f}x")
    ax1.legend(loc='upper right')

    # AER Plot
    ax2.plot(res['aer_times'], res['aer_mean'], color='darkblue', linewidth=1.5)
    ax2.axvline(0, color='red', linestyle='--', alpha=0.3)
    ax2.set_ylabel('Source Amp (nAm)')
    ax2.set_xlabel('Time relative to Cortical Spike (ms)')
    ax2.set_title(f'Thalamic Average Evoked Response (n={res["n_spikes"]})')

    plt.tight_layout()
    # plt.show()  # 注释掉show以避免阻塞，改为保存图片
    plt.savefig(os.path.join(save_dir, f"{tag}_ccg_aer.png"), dpi=150, bbox_inches='tight')
    plt.close()