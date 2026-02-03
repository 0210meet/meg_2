#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
04_SOZ_Based_CCG_Analysis.py

基于SOZ的癫痫灶-丘脑CCG分析

两个关键改进：
1. 使用SOZ（癫痫灶）信号替代最活跃皮层区域
2. 只分析明确确定的高置信度癫痫放电

基于Wodeyar et al. (2024)方法
"""

import os
import mne
import numpy as np
from scipy.signal import find_peaks, hilbert
from mne.filter import filter_data
import matplotlib.pyplot as plt

# ===================== 配置参数 =====================
SUBJECTS = ["sub-01"]  # 测试
STATES = ["EC", "EO"]

SOURCE_DIR = "/data/shared_home/tlm/Project/MEG-C/source"
FREESURFER_DIR = "/data/shared_home/tlm/data/MEG-C/freesurfer/mri"
SAVE_DIR = "/data/shared_home/tlm/Project/MEG-C/results_SOZ"
os.makedirs(SAVE_DIR, exist_ok=True)

# Wodeyar 2024参数
IED_BAND = (25, 80)
THRESHOLD_MEAN_MULT = 3
FANO_FACTOR_THRESH = 2.5
MIN_INTERVAL_MS = 20

# SOZ检测参数
SOZ_PERCENTILE_THRESH = 95  # 95th percentile作为SOZ阈值
SOZ_MIN_CLUSTER_SIZE = 10   # 最小聚类大小

# 高置信度IED参数（更严格）
HIGH_CONFIDENCE_Z_THRESH = 5.0  # 更高的Z阈值
HIGH_CONFIDENCE_MIN_AMP = np.percentile  # 排除最低幅值

# ===================== SOZ检测函数 =====================
def identify_soz_from_stc(stc, percentile_thresh=95):
    """
    基于源定位强度识别癫痫灶(SOZ)
    
    原理：癫痫灶通常显示持续的高幅值异常活动
    """
    # 1. 计算每个源点的平均激活强度
    source_power = np.mean(np.abs(stc.data), axis=1)
    
    # 2. 设定阈值
    threshold = np.percentile(source_power, percentile_thresh)
    
    # 3. 识别超过阈值的源点索引
    soz_sources = np.where(source_power > threshold)[0]
    
    print(f"    SOZ检测: {len(soz_sources)}/{len(source_power)} 源点超过{percentile_thresh}th percentile")
    
    return soz_sources, source_power, threshold


def create_soz_label_from_sources(src, soz_source_indices, hemi):
    """
    从源点索引创建SOZ label
    """
    from mne import Label
    
    # 获取该半球的所有顶点
    if hemi == 'lh':
        src_idx = 0
    else:
        src_idx = 1
    
    all_vertices = src[src_idx]['vertno']
    
    # 找到soz源点对应的顶点号
    soz_vertices = []
    for src_idx_in_soz in soz_source_indices:
        if src_idx_in_soz < len(all_vertices):
            soz_vertices.append(all_vertices[src_idx_in_soz])
    
    if len(soz_vertices) == 0:
        return None
    
    soz_label = Label(vertices=soz_vertices, hemi=hemi, 
                     name=f"SOZ-{hemi}", subject=src[0]['subject'])
    
    return soz_label


def detect_ied_wodeyar_with_confidence(sig_raw, sfreq, high_confidence=False):
    """
    IED检测（带置信度分级）
    
    Args:
        high_confidence: 是否只返回高置信度IED
    """
    # 1. 滤波
    sig_filt = filter_data(
        sig_raw.astype(float), sfreq,
        IED_BAND[0], IED_BAND[1],
        method='fir', phase='zero-double', verbose=False
    )
    
    # 2. Hilbert包络
    env = np.abs(hilbert(sig_filt))
    env_z = zscore(env)
    
    # 3. 阈值检测
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
    
    # 6. 高置信度筛选（可选）
    if high_confidence:
        # 只保留Z-score > 5的spikes
        high_conf_indices = np.where(env_z[merged] > HIGH_CONFIDENCE_Z_THRESH)[0]
        merged = merged[high_conf_indices]
        
        # 排除最低幅值的20%
        if len(merged) > 5:
            amplitudes = env[merged]
            amp_thresh = np.percentile(amplitudes, 20)
            merged = merged[amplitudes >= amp_thresh]
    
    return merged


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


def compute_ccg_simple(ref_times, target_times, sfreq, window_sec=1.0):
    """
    简化的CCG计算（用于高置信度spikes）
    """
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
    
    # 简单的shuffle检验（1000次，高置信度spikes较少）
    N_SHUFFLE = 1000
    null_rates = []
    
    for _ in range(N_SHUFFLE):
        shuf_ref = np.random.uniform(0, ref_times[-1], size=len(ref_times))
        shuf_diffs = []
        for sr in shuf_ref:
            matches = target_times[np.abs(target_times - sr) <= window_sec]
            shuf_diffs.extend(matches - sr)
        
        h, _ = np.histogram(shuf_diffs, bins=bins)
        null_rates.append(h / len(ref_times) / bin_sec)
    
    upper_95 = np.percentile(null_rates, 97.5, axis=0)
    
    # 计算统计
    mask_post = (bin_centers > 0) & (bin_centers <= 0.5)
    rate_post_sum = np.sum(rate[mask_post])
    
    # 潜伏期
    sig_mask = rate > upper_95
    latencies = bin_centers[sig_mask & (bin_centers > 0)]
    latency = latencies[0] * 1000 if len(latencies) > 0 else None
    
    # 增强
    base_rate = np.mean(rate[bin_centers < 0]) + 1e-6
    enhancement = np.max(rate[mask_post]) / base_rate if any(mask_post) else 0
    
    return {
        'centers': bin_centers * 1000,
        'rate': rate,
        'upper': upper_95,
        'latency': latency,
        'enhancement': enhancement,
        'sig_mask': sig_mask,
        'n_ref': len(ref_times),
        'n_target': len(target_times)
    }


# ===================== 主处理函数 =====================
def process_with_soz(subject, run, state, src):
    """使用SOZ进行处理"""
    try:
        # 1. 读取STC
        stc_fname = os.path.join(SOURCE_DIR, subject, run, 
                                   f"{subject}-{run}-{state}-mixed-stc.h5")
        if not os.path.exists(stc_fname):
            print(f"    ❌ STC文件不存在")
            return False
        
        stc = mne.read_source_estimate(stc_fname)
        sfreq = 1 / (stc.times[1] - stc.times[0])
        
        # 2. 识别SOZ
        print(f"\n  [SOZ检测]")
        soz_sources, source_power, threshold = identify_soz_from_stc(
            stc, percentile_thresh=SOZ_PERCENTILE_THRESH
        )
        
        if len(soz_sources) == 0:
            print(f"    ❌ 未检测到SOZ")
            return False
        
        # 3. 创建SOZ label
        # 判断优势半球
        left_power = np.sum(source_power[:stc.data.shape[0]//2])
        right_power = np.sum(source_power[stc.data.shape[0]//2:])
        dominant_hemi = 'lh' if left_power > right_power else 'rh'
        
        soz_label = create_soz_label_from_sources(src, soz_sources, dominant_hemi)
        
        if soz_label is None:
            print(f"    ❌ 创建SOZ label失败")
            return False
        
        print(f"    ✅ SOZ创建成功: {soz_label.name} ({len(soz_label.vertices)} vertices)")
        
        # 4. 提取SOZ时间序列
        soz_ts = mne.extract_label_time_course(
            [stc], [soz_label], src, mode="mean", allow_empty=True
        )[0][0]
        
        # 5. 检测SOZ中的IED（高置信度）
        print(f"\n  [IED检测 - 高置信度]")
        soz_z = zscore(soz_ts)
        soz_spikes_all = detect_ied_wodeyar_with_confidence(soz_ts, sfreq, high_confidence=False)
        soz_spikes_high_conf = detect_ied_wodeyar_with_confidence(soz_ts, sfreq, high_confidence=True)
        
        print(f"    总spikes: {len(soz_spikes_all)}")
        print(f"    高置信度spikes: {len(soz_spikes_high_conf)}")
        
        if len(soz_spikes_high_conf) < 3:
            print(f"    ❌ 高置信度spikes数量不足")
            return False
        
        # 6. 构建同侧丘脑ROI
        thal_config = {"left": {"src_idx": 2, "name": "Left-Thalamus-Proper", "hemi": "lh"},
                        "right": {"src_idx": 3, "name": "Right-Thalamus-Proper", "hemi": "rh"}}[dominant_hemi == 'lh' and 0 or 1]
        
        thal_label = mne.Label(
            vertices=src[thal_config["src_idx"]]["vertno"],
            hemi=thal_config["hemi"],
            name=thal_config["name"],
            subject=subject
        )
        
        thal_ts = mne.extract_label_time_course(
            [stc], [thal_label], src, mode="mean", allow_empty=True
        )[0][0]
        
        # 7. 检测丘脑IED
        thal_spikes_all = detect_ied_wodeyar_with_confidence(thal_ts, sfreq, high_confidence=False)
        
        print(f"    丘脑spikes: {len(thal_spikes_all)}")
        
        # 8. CCG分析（只使用高置信度SOZ spikes）
        print(f"\n  [CCG分析: SOZ→丘脑]")
        ccg_results = compute_ccg_simple(
            soz_spikes_high_conf / sfreq,
            thal_spikes_all / sfreq,
            sfreq
        )
        
        tag = f"{subject}_{run}_{state}_SOZ"
        case_save_dir = os.path.join(SAVE_DIR, subject, run, state)
        os.makedirs(case_save_dir, exist_ok=True)
        
        np.save(os.path.join(case_save_dir, f"{tag}_ccg_results.npy"), ccg_results)
        
        print(f"\n  结果:")
        print(f"    传播方向: SOZ→丘脑")
        print(f"    潜伏期: {ccg_results['latency']} ms" if ccg_results['latency'] else f"    潜伏期: N/A")
        print(f"    信号增强: {ccg_results['enhancement']:.2f}x")
        print(f"    SOZ spike数: {ccg_results['n_ref']}")
        print(f"    丘脑spike数: {ccg_results['n_target']}")
        
        return True
        
    except Exception as e:
        print(f"    ❌ 处理失败: {str(e)[:100]}")
        return False


# ===================== 主程序 =====================
if __name__ == "__main__":
    print("=" * 80)
    print("基于SOZ的癫痫灶-丘脑CCG分析")
    print("=" * 80)
    
    for subject in SUBJECTS:
        print(f"\n{'='*60}\n处理被试: {subject}\n{'='*60}")
        
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
                print(f"\n[{subject}-{run}-{state}]")
                process_with_soz(subject, run, state, src)
        
        del src
    
    print("\n" + "=" * 80)
    print("✅ SOZ-based分析完成！")
    print(f"结果保存至: {SAVE_DIR}")
    print("=" * 80)
