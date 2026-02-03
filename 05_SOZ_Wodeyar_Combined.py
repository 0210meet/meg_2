#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
05_SOZ_Wodeyar_Combined.py

结合Wodeyar 2024文献方法的SOZ-based CCG分析

两个优势结合：
1. SOZ：使用癫痫灶信号（而非最活跃ROI）
2. Wodeyar方法：简化IED检测 + Fano factor + 严格统计

关键特性：
- SOZ识别（95th percentile）
- Wodeyar标准IED检测（25-80Hz, 3×mean, Fano>2.5）
- 高置信度spike筛选（Z>5）
- ±1s分析窗口
- 完整的三重验证（AER + CCH）
"""

import os
import mne
import numpy as np
from scipy.signal import find_peaks, hilbert
from scipy.stats import zscore
from mne.filter import filter_data
import matplotlib.pyplot as plt

# ===================== 配置参数 =====================
SUBJECTS = [f"sub-{i:02d}" for i in range(1, 9)]  # 全部8个被试
STATES = ["EC", "EO"]

SOURCE_DIR = "/data/shared_home/tlm/Project/MEG-C/source_DBA"  # 使用DBA源定位结果
FREESURFER_DIR = "/data/shared_home/tlm/data/MEG-C/freesurfer/mri"
SAVE_DIR = "/data/shared_home/tlm/Project/MEG-C/results_SOZ_Wodeyar"
os.makedirs(SAVE_DIR, exist_ok=True)

# Wodeyar 2024参数（文献标准）
IED_BAND = (25, 80)  # 文献使用25-80Hz
THRESHOLD_MEAN_MULT = 3  # 3倍平均幅度
FANO_FACTOR_THRESH = 2.5
MIN_INTERVAL_MS = 20

# SOZ参数
SOZ_PERCENTILE_THRESH = 95

# 高置信度参数
HIGH_CONFIDENCE_Z_THRESH = 5.0

# 分析参数（文献标准）
AER_WINDOW_SEC = 1.0  # ±1s
CCH_WINDOW_SEC = 1.0
CCH_BIN_SEC = 0.035  # ~35ms bins
N_SHUFFLE = 5000  # 5000次shuffle

# ===================== 工具函数 =====================
def identify_soz_from_stc(stc, percentile_thresh=95):
    """
    基于源定位强度识别SOZ
    """
    source_power = np.mean(np.abs(stc.data), axis=1)
    threshold = np.percentile(source_power, percentile_thresh)
    soz_sources = np.where(source_power > threshold)[0]
    
    print(f"    SOZ检测: {len(soz_sources)}/{len(source_power)} 源点超过{percentile_thresh}th percentile")
    print(f"    SOZ平均功率: {np.mean(source_power[soz_sources]):.2f}")
    print(f"    全局平均功率: {np.mean(source_power):.2f}")
    
    return soz_sources, source_power, threshold


def create_soz_label_from_sources(src, soz_source_indices, hemi):
    """从源点创建SOZ label"""
    from mne import Label

    # 混合源空间: [lh_cortex, rh_cortex, lh_thalamus, rh_thalamus, ...其他volume源]
    # 对于皮层SOZ，我们只使用前两个源空间（皮层）
    if hemi == 'lh':
        src_idx = 0  # 左半球皮层
    else:
        src_idx = 1  # 右半球皮层

    all_vertices = src[src_idx]['vertno']

    soz_vertices = []
    for src_idx_in_soz in soz_source_indices:
        # 确保索引在有效范围内
        if 0 <= src_idx_in_soz < len(all_vertices):
            soz_vertices.append(all_vertices[src_idx_in_soz])

    if len(soz_vertices) == 0:
        return None

    soz_label = Label(vertices=soz_vertices, hemi=hemi,
                     name=f"SOZ-Wodeyar-{hemi}")
    return soz_label


def fano_factor(signal):
    """计算Fano factor"""
    peaks, _ = find_peaks(signal)
    troughs, _ = find_peaks(-signal)
    
    if len(peaks) < 2 or len(troughs) < 2:
        return 0
    
    peak_intervals = np.diff(peaks)
    trough_intervals = np.diff(troughs)
    all_intervals = np.concatenate([peak_intervals, trough_intervals])
    
    if len(all_intervals) == 0:
        return 0
    
    return np.var(all_intervals) / np.mean(all_intervals)


def detect_ied_wodeyar(sig_raw, sfreq, high_confidence=False):
    """
    Wodeyar 2024标准IED检测
    
    Args:
        high_confidence: 是否只返回高置信度IED (Z>5)
    """
    # 1. 25-80Hz FIR滤波
    sig_filt = filter_data(
        sig_raw.astype(float), sfreq,
        IED_BAND[0], IED_BAND[1],
        method='fir', phase='zero-double', verbose=False
    )
    
    # 2. Hilbert包络
    env = np.abs(hilbert(sig_filt))
    env_z = zscore(env)
    
    # 3. 阈值检测：3×mean
    thresh = THRESHOLD_MEAN_MULT * np.mean(env)
    candidate_peaks, _ = find_peaks(env, height=thresh)
    
    if len(candidate_peaks) == 0:
        return np.array([], dtype=int)
    
    # 4. Fano factor + 幅值检验
    keep = []
    for cp in candidate_peaks:
        w = int(0.25 * sfreq)
        if cp - w < 0 or cp + w >= len(sig_raw):
            continue
        
        segment = sig_raw[cp-w:cp+w]
        ff = fano_factor(segment)
        max_amp = np.max(np.abs(segment))
        mean_amp = np.mean(np.abs(sig_raw))
        
        if ff > FANO_FACTOR_THRESH and max_amp > THRESHOLD_MEAN_MULT * mean_amp:
            keep.append(cp)
    
    if len(keep) == 0:
        return np.array([], dtype=int)
    
    keep = np.array(keep, dtype=int)
    
    # 5. 合并20ms内的spikes
    min_dist = int(MIN_INTERVAL_MS * sfreq / 1000)
    merged = []
    groups = [[keep[0]]]
    
    for p in keep[1:]:
        if p - groups[-1][-1] <= min_dist:
            groups[-1].append(p)
        else:
            groups.append([p])
    
    for g in groups:
        g_env = env[g]
        merged.append(g[np.argmax(g_env)])
    
    merged = np.array(sorted(merged), dtype=int)
    
    # 6. 高置信度筛选
    if high_confidence:
        high_conf_indices = np.where(env_z[merged] > HIGH_CONFIDENCE_Z_THRESH)[0]
        merged = merged[high_conf_indices]
        
        if len(merged) > 5:
            amplitudes = env[merged]
            amp_thresh = np.percentile(amplitudes, 20)
            merged = merged[amplitudes >= amp_thresh]
    
    return merged


def compute_aer(ref_times, target_sig, sfreq, window_sec=1.0):
    """Average Evoked Response"""
    half_win = int(window_sec * sfreq)

    epochs = []
    for spike_time in ref_times:
        # 确保spike_time是整数类型
        spike_idx = int(spike_time)
        if spike_idx - half_win >= 0 and spike_idx + half_win < len(target_sig):
            epoch = target_sig[spike_idx - half_win: spike_idx + half_win].copy()
            baseline_len = int(0.2 * len(epoch))
            epoch -= np.mean(epoch[:baseline_len])
            epochs.append(epoch)

    if len(epochs) == 0:
        return None, None, None

    aer_mean = np.mean(epochs, axis=0)
    aer_sem = np.std(epochs, axis=0) / np.sqrt(len(epochs))
    times = np.linspace(-window_sec, window_sec, len(aer_mean))

    return times, aer_mean, aer_sem


def compute_ccg(ref_times, target_times, window_sec=1.0):
    """Cross-Correlation Histogram with shuffle检验"""
    bin_sec = 0.01  # 10ms bins
    bins = np.arange(-window_sec, window_sec + bin_sec, bin_sec)
    bin_centers = (bins[:-1] + bins[1:]) / 2
    
    # 计算CCG
    diffs = []
    for rt in ref_times:
        diffs.extend(target_times - rt)
    
    diffs = np.array(diffs)
    diffs = diffs[np.abs(diffs) <= window_sec]
    
    hist, _ = np.histogram(diffs, bins=bins)
    rate = hist / len(ref_times) / bin_sec
    
    # Shuffle检验（5000次）
    null_rates = []
    for _ in range(N_SHUFFLE):
        shuf_ref = np.random.uniform(0, max(ref_times[-1], 1), size=len(ref_times))
        shuf_diffs = []
        for sr in shuf_ref:
            matches = target_times[np.abs(target_times - sr) <= window_sec]
            shuf_diffs.extend(matches - sr)
        h, _ = np.histogram(shuf_diffs, bins=bins)
        null_rates.append(h / len(ref_times) / bin_sec)
    
    null_rates = np.array(null_rates)
    upper_95 = np.percentile(null_rates, 97.5, axis=0)
    
    # 统计
    mask_post = (bin_centers > 0) & (bin_centers <= 0.5)
    rate_post_sum = np.sum(rate[mask_post])
    
    sig_mask = rate > upper_95
    latencies = bin_centers[sig_mask & (bin_centers > 0)]
    latency = latencies[0] * 1000 if len(latencies) > 0 else None
    
    base_rate = np.mean(rate[bin_centers < 0]) + 1e-6
    enhancement = np.max(rate[mask_post]) / base_rate if any(mask_post) else 0
    
    # DI指数（8-40ms窗口）
    mask_di_post = (bin_centers >= 0.008) & (bin_centers <= 0.040)
    mask_di_pre = (bin_centers <= -0.008) & (bin_centers >= -0.040)
    rate_post_sum = np.sum(rate[mask_di_post])
    rate_pre_sum = np.sum(rate[mask_di_pre])
    di = (rate_post_sum - rate_pre_sum) / (rate_post_sum + rate_pre_sum + 1e-6)
    
    return {
        'centers': bin_centers * 1000,
        'rate': rate,
        'upper': upper_95,
        'latency': latency,
        'enhancement': enhancement,
        'sig_mask': sig_mask,
        'di': di,
        'n_ref': len(ref_times),
        'n_target': len(target_times)
    }


# ===================== 主处理函数 =====================
def process_soz_wodeyar(subject, run, state, src):
    """结合SOZ和Wodeyar方法"""
    try:
        # 1. 读取STC
        stc_fname = os.path.join(SOURCE_DIR, subject, run,
                                   f"{subject}-{run}-{state}-DBA-dSPM-stc.h5")
        if not os.path.exists(stc_fname):
            return False
        
        stc = mne.read_source_estimate(stc_fname)
        sfreq = 1 / (stc.times[1] - stc.times[0])
        
        print(f"\n  {'='*50}")
        print(f"  {subject}-{run}-{state}")
        print(f"  {'='*50}")
        
        # 2. 识别SOZ（结合Wodeyar方法）
        print(f"\n  [1] SOZ识别 (95th percentile)")
        soz_sources, source_power, threshold = identify_soz_from_stc(
            stc, percentile_thresh=SOZ_PERCENTILE_THRESH
        )
        
        if len(soz_sources) == 0:
            print(f"    ❌ 未检测到SOZ")
            return False
        
        # 3. 创建SOZ label
        left_power = np.sum(source_power[:stc.data.shape[0]//2])
        right_power = np.sum(source_power[stc.data.shape[0]//2:])
        dominant_hemi = 'lh' if left_power > right_power else 'rh'
        
        soz_label = create_soz_label_from_sources(src, soz_sources, dominant_hemi)
        
        if soz_label is None:
            print(f"    ❌ 创建SOZ label失败")
            return False
        
        print(f"    ✅ SOZ: {soz_label.name} ({len(soz_label.vertices)} vertices)")
        
        # 4. 提取SOZ时间序列
        soz_ts = mne.extract_label_time_course(
            [stc], [soz_label], src, mode="mean", allow_empty=True
        )[0][0]
        
        # 5. Wodeyar标准IED检测
        print(f"\n  [2] IED检测 (Wodeyar 2024标准)")
        print(f"    频带: {IED_BAND[0]}-{IED_BAND[1]} Hz")
        print(f"    阈值: {THRESHOLD_MEAN_MULT}×mean")
        print(f"    Fano factor: >{FANO_FACTOR_THRESH}")
        
        soz_spikes_all = detect_ied_wodeyar(soz_ts, sfreq, high_confidence=False)
        soz_spikes_high_conf = detect_ied_wodeyar(soz_ts, sfreq, high_confidence=True)
        
        print(f"    总spikes: {len(soz_spikes_all)}")
        print(f"    高置信度spikes (Z>{HIGH_CONFIDENCE_Z_THRESH}): {len(soz_spikes_high_conf)}")
        
        if len(soz_spikes_high_conf) < 3:
            print(f"    ❌ 高置信度spikes数量不足")
            return False
        
        # 6. 同侧丘脑
        # 根据dominant_hemi选择正确的丘脑配置
        thal_key = "left" if dominant_hemi == 'lh' else "right"
        thal_config_map = {
            "left": {"src_idx": 2, "name": "Left-Thalamus-Proper", "hemi": "lh"},
            "right": {"src_idx": 3, "name": "Right-Thalamus-Proper", "hemi": "rh"}
        }
        thal_config = thal_config_map[thal_key]
        
        thal_label = mne.Label(
            vertices=src[thal_config["src_idx"]]["vertno"],
            hemi=thal_config["hemi"],
            name=thal_config["name"]
        )
        
        thal_ts = mne.extract_label_time_course(
            [stc], [thal_label], src, mode="mean", allow_empty=True
        )[0][0]
        
        thal_spikes = detect_ied_wodeyar(thal_ts, sfreq, high_confidence=False)

        print(f"    丘脑spikes: {len(thal_spikes)}")

        # 检查丘脑spikes数量
        if len(thal_spikes) < 3:
            print(f"    ❌ 丘脑spikes数量不足，跳过分析")
            return False

        # 7. AER分析（±1s窗口，文献标准）
        print(f"\n  [3] AER分析 (±{AER_WINDOW_SEC}s窗口)")
        times, aer_mean, aer_sem = compute_aer(
            soz_spikes_high_conf,  # 直接传递样本索引，不除以sfreq
            thal_ts,
            sfreq,
            window_sec=AER_WINDOW_SEC
        )
        
        if aer_mean is not None:
            print(f"    ✅ AER计算完成 (n={len(soz_spikes_high_conf)})")
            # 检查0-1s窗口内丘脑响应是否显著增加
            half_idx = len(aer_mean) // 2
            post_response = np.mean(np.abs(aer_mean[half_idx:]))
            pre_response = np.mean(np.abs(aer_mean[:half_idx]))
            print(f"    触发前平均幅值: {pre_response:.4f}")
            print(f"    触发后平均幅值: {post_response:.4f}")
            print(f"    传播效应: {(post_response/pre_response - 1)*100:.1f}%")
        
        # 8. CCG分析（±1s，5000次shuffle）
        print(f"\n  [4] CCG分析 (±{CCH_WINDOW_SEC}s, {N_SHUFFLE}次shuffle)")
        ccg_results = compute_ccg(
            soz_spikes_high_conf / sfreq,
            thal_spikes / sfreq,
            window_sec=CCH_WINDOW_SEC
        )
        
        print(f"    潜伏期: {ccg_results['latency']} ms" if ccg_results['latency'] else f"    潜伏期: N/A")
        print(f"    信号增强: {ccg_results['enhancement']:.2f}x")
        print(f"    方向性指数(DI): {ccg_results['di']:.3f}")
        direction = "SOZ→丘脑" if ccg_results['di'] > 0 else "丘脑→SOZ"
        print(f"    传播方向: {direction}")
        
        # 9. 保存结果
        tag = f"{subject}_{run}_{state}"
        case_save_dir = os.path.join(SAVE_DIR, subject, run, state)
        os.makedirs(case_save_dir, exist_ok=True)
        
        np.save(os.path.join(case_save_dir, f"{tag}_soz_wodeyar.npy"), ccg_results)
        
        # 10. 绘图
        fig, axes = plt.subplots(2, 1, figsize=(10, 8))
        
        # AER
        if aer_mean is not None:
            axes[0].plot(times * 1000, aer_mean, color='steelblue', linewidth=1.5, label='AER')
            axes[0].fill_between(times * 1000, 
                                 aer_mean - 1.96*aer_sem, 
                                 aer_mean + 1.96*aer_sem, 
                                 color='steelblue', alpha=0.3)
            axes[0].axvline(0, color='red', linestyle='--', alpha=0.5)
            axes[0].axhline(0, color='black', linewidth=0.5)
            axes[0].set_ylabel('Amplitude (a.u.)')
            axes[0].set_title(f'AER: SOZ→Thalamus (n={len(soz_spikes_high_conf)} spikes)')
            axes[0].legend()
            axes[0].grid(True, alpha=0.3)
        
        # CCG
        axes[1].bar(ccg_results['centers'], ccg_results['rate'], 
                   width=10, color='steelblue', alpha=0.6, label='CCG')
        axes[1].fill_between(ccg_results['centers'], 
                             ccg_results['rate'], 
                             ccg_results['upper'], 
                             color='gray', alpha=0.3, label='95% CI')
        axes[1].axvline(0, color='red', linestyle='--', alpha=0.5)
        axes[1].set_xlabel('Lag (ms)')
        axes[1].set_ylabel('Rate (Hz)')
        axes[1].set_title(f'CCG: DI={ccg_results["di"]:.3f} ({direction})')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(case_save_dir, f"{tag}_soz_wodeyar.png"), 
                   dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"    ✅ 结果保存至: {case_save_dir}")
        
        return True
        
    except Exception as e:
        print(f"    ❌ 处理失败: {str(e)[:100]}")
        import traceback
        traceback.print_exc()
        return False


# ===================== 主程序 =====================
if __name__ == "__main__":
    print("=" * 80)
    print("SOZ + Wodeyar 2024 方法结合分析")
    print("=" * 80)
    print("\n结合两个优势：")
    print("1. SOZ: 使用癫痫灶信号（而非最活跃ROI）")
    print("2. Wodeyar 2024: 简化IED检测 + Fano factor + 严格统计")
    print("=" * 80)
    
    for subject in SUBJECTS:
        print(f"\n{'='*60}\n被试: {subject}\n{'='*60}")
        
        # 加载源空间
        bem_dir = os.path.join(FREESURFER_DIR, subject, "bem")
        mixed_src_fname = os.path.join(bem_dir, f"{subject}-mixed-src.fif")
        
        if not os.path.exists(mixed_src_fname):
            print(f"❌ 混合源空间不存在")
            continue
        
        src = mne.read_source_spaces(mixed_src_fname)
        print(f"✅ 加载混合源空间")
        
        # 获取runs
        subj_source_dir = os.path.join(SOURCE_DIR, subject)
        run_dirs = sorted([d for d in os.listdir(subj_source_dir)
                           if os.path.isdir(os.path.join(subj_source_dir, d)) 
                           and d.startswith("run-")])
        
        if not run_dirs:
            print(f"❌ 无有效run目录")
            continue
        
        print(f"✅ 检测到runs: {run_dirs}")
        
        for run in run_dirs:
            for state in STATES:
                process_soz_wodeyar(subject, run, state, src)
        
        del src
    
    print("\n" + "=" * 80)
    print("✅ SOZ + Wodeyar分析完成！")
    print(f"结果保存至: {SAVE_DIR}")
    print("=" * 80)
