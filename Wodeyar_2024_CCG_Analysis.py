#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Wodeyar_2024_CCG_Analysis.py

基于Wodeyar et al. (2024) Brain 147:2803-2816的丘脑-皮层CCG分析

文献结论："cortical epileptic spikes propagate to the thalamus"
本方法完全遵循文献的分析流程

关键特性：
1. 简化的IED检测（无形态学约束）
2. Fano factor检验
3. 三种互补方法：AER + CCH + GLM
4. 严格的统计检验（5000次shuffle）
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks, hilbert, filtfilt, fir_filter_design
from scipy.stats import zscore, percentileofscore
from mne.filter import filter_data
from statsmodels.api import GLM, families
import pandas as pd

# ===================== 文献方法参数 =====================
# Wodeyar et al. (2024) 标准参数
IED_BAND = (25, 80)  # 文献使用25-80Hz
THRESHOLD_MEAN_MULT = 3  # 3倍平均幅度（文献标准）
FANO_FACTOR_THRESH = 2.5  # Fano factor阈值
MIN_INTERVAL_MS = 20  # 20ms内合并

# 分析参数
AER_WINDOW_SEC = 1.0  # AER窗口：±1s
CCH_WINDOW_SEC = 1.0  # CCH窗口：±1s
CCH_BIN_SEC = 0.035  # ~35ms bins（文献标准）
N_SHUFFLE = 5000  # 5000次shuffle（文献标准）
GLM_BIN_SEC = 0.016  # 16ms bins（文献标准）

# ===================== 工具函数 =====================
def fano_factor(signal):
    """计算Fano factor（方差/均值比）
    
    Fano factor < 1: 规律性振荡（如gamma burst）
    Fano factor > 1: 不规律事件
    文献使用 > 2.5 来排除gamma burst
    """
    if len(signal) == 0:
        return 0
    # 检测峰和谷
    peaks, _ = find_peaks(signal)
    troughs, _ = find_peaks(-signal)
    
    if len(peaks) < 2 or len(troughs) < 2:
        return 0
    
    # 计算峰间和谷间间隔
    peak_intervals = np.diff(peaks)
    trough_intervals = np.diff(troughs)
    
    all_intervals = np.concatenate([peak_intervals, trough_intervals])
    
    if len(all_intervals) == 0:
        return 0
    
    # Fano factor = 方差/均值
    ff = np.var(all_intervals) / np.mean(all_intervals)
    return ff


def detect_ied_wodeyar(sig_raw, sfreq):
    """
    基于Wodeyar et al. (2024)的IED检测方法
    
    关键：无形态学约束，仅使用包络和Fano factor
    """
    # 1. 25-80Hz FIR滤波（双向滤波）
    sig_filt = filter_data(
        sig_raw.astype(float), sfreq,
        IED_BAND[0], IED_BAND[1],
        method='fir',
        phase='zero-double',  # 双向滤波
        verbose=False
    )
    
    # 2. Hilbert变换获取包络
    env = np.abs(hilbert(sig_filt))
    
    # 3. 阈值检测：3倍平均幅度
    thresh = THRESHOLD_MEAN_MULT * np.mean(env)
    candidate_peaks, _ = find_peaks(env, height=thresh)
    
    print(f"  候选peaks: {len(candidate_peaks)}, 阈值: {thresh:.2f}")
    
    if len(candidate_peaks) == 0:
        return np.array([], dtype=int)
    
    # 4. Fano factor检验（排除gamma burst）
    keep = []
    for cp in candidate_peaks:
        # 检查是否在信号范围内
        w = int(0.25 * sfreq)
        if cp - w < 0 or cp + w >= len(sig_raw):
            continue
        
        # 提取±0.25s窗口
        segment = sig_raw[cp-w:cp+w]
        
        # 计算Fano factor
        ff = fano_factor(segment)
        
        # 检查最大幅度（排除低幅值候选）
        max_amp = np.max(np.abs(segment))
        mean_amp = np.mean(np.abs(sig_raw))
        
        # 文献标准：
        # 1. Fano factor > 2.5
        # 2. 最大幅度 > 3×平均幅度
        if ff > FANO_FACTOR_THRESH and max_amp > THRESHOLD_MEAN_MULT * mean_amp:
            keep.append(cp)
    
    keep = np.array(keep, dtype=int)
    
    if len(keep) == 0:
        return keep
    
    # 5. 合并20ms内的spikes
    # 找到间距小于20ms的峰
    min_dist = int(MIN_INTERVAL_MS * sfreq / 1000)
    merged = []
    groups = [[keep[0]]]
    
    for p in keep[1:]:
        if p - groups[-1][-1] <= min_dist:
            groups[-1].append(p)
        else:
            groups.append([p])
    
    # 每组保留幅度最大的
    for g in groups:
        g_env = env[g]
        merged.append(g[np.argmax(g_env)])
    
    merged = np.array(sorted(merged), dtype=int)
    
    print(f"  最终spikes: {len(merged)} (Fano factor > {FANO_FACTOR_THRESH})")
    
    return merged


def compute_aer_wodeyar(ref_times, target_sig, sfreq):
    """
    Average Evoked Response (AER)
    文献方法：±1s窗口，皮层spike触发，观察丘脑响应
    """
    half_win = int(AER_WINDOW_SEC * sfreq)
    
    epochs = []
    for spike_time in ref_times:
        if spike_time - half_win >= 0 and spike_time + half_win < len(target_sig):
            epoch = target_sig[spike_time - half_win: spike_time + half_win].copy()
            # 基线校正：前20%窗口
            baseline_len = int(0.2 * len(epoch))
            epoch -= np.mean(epoch[:baseline_len])
            epochs.append(epoch)
    
    if len(epochs) == 0:
        return None, None
    
    aer_mean = np.mean(epochs, axis=0)
    aer_sem = np.std(epochs, axis=0) / np.sqrt(len(epochs))
    times = np.linspace(-AER_WINDOW_SEC, AER_WINDOW_SEC, len(aer_mean))
    
    return times, aer_mean, aer_sem


def compute_ccg_wodeyar(ref_times, target_times):
    """
    Cross-Correlation Histogram (CCH)
    文献方法：未归一化，±1s窗口，~35ms bins
    """
    bins = np.arange(-CCH_WINDOW_SEC, CCH_WINDOW_SEC + CCH_BIN_SEC, CCH_BIN_SEC)
    bin_centers = (bins[:-1] + bins[1:]) / 2
    
    # 计算原始CCG
    diffs = []
    for rt in ref_times:
        diffs.extend(target_times - rt)
    
    diffs = np.array(diffs)
    diffs = diffs[np.abs(diffs) <= CCH_WINDOW_SEC]
    
    hist, _ = np.histogram(diffs, bins=bins)
    
    # 5000次shuffle检验（文献标准）
    print(f"  进行{N_SHUFFLE}次shuffle检验...")
    null_dist = []
    
    for _ in range(N_SHUFFLE):
        # Shuffle inter-event intervals（文献方法）
        # 随机化spike之间的间隔，保留间隔分布
        if len(ref_times) > 1:
            intervals = np.diff(ref_times)
            np.random.shuffle(intervals)
            shuf_times = np.concatenate([[0], np.cumsum(intervals)])
            # 确保不超出范围
            shuf_times = shuf_times[shuf_times <= ref_times[-1]]
        else:
            shuf_times = np.random.uniform(0, ref_times[0], size=len(ref_times))
        
        # 计算shuffled CCG
        shuf_diffs = []
        for st in shuf_times:
            shuf_diffs.extend(target_times - st)
        
        shuf_diffs = np.array(shuf_diffs)
        shuf_diffs = shuf_diffs[np.abs(shuf_diffs) <= CCH_WINDOW_SEC]
        
        h, _ = np.histogram(shuf_diffs, bins=bins)
        null_dist.append(h)
    
    null_dist = np.array(null_dist)
    
    # 95%置信区间
    ci_low = np.percentile(null_dist, 2.5, axis=0)
    ci_high = np.percentile(null_dist, 97.5, axis=0)
    
    # Bonferroni校正（文献方法）
    alpha_corrected = 0.05 / len(bins)
    ci_low_bonf = np.percentile(null_dist, alpha_corrected/2*100, axis=0)
    ci_high_bonf = np.percentile(null_dist, (1-alpha_corrected/2)*100, axis=0)
    
    return bin_centers * 1000, hist, ci_low, ci_high, ci_low_bonf, ci_high_bonf


def fit_glm_point_process(ref_times, target_times, signal_length, sfreq):
    """
    Point Process Model (GLM)
    文献方法：16ms bins，Poisson GLM
    
    log λ(t) = β0 + Σγ·ΔN_cortical(t-i) + Σβ·ΔN_thalamic(t-i)
    """
    bin_size = int(GLM_BIN_SEC * sfreq)
    n_bins = int(signal_length / bin_size)
    
    # 创建二值时间序列
    ref_ts = np.zeros(n_bins)
    target_ts = np.zeros(n_bins)
    
    for rt in ref_times:
        bin_idx = int(rt / bin_size)
        if 0 <= bin_idx < n_bins:
            ref_ts[bin_idx] += 1
    
    for tt in target_times:
        bin_idx = int(tt / bin_size)
        if 0 <= bin_idx < n_bins:
            target_ts[bin_idx] += 1
    
    # 创建设计矩阵（自回归 + 交叉项）
    max_lags = 20  # 文献使用20个lag
    
    X = []
    y = target_ts[max_lags:]
    
    for i in range(max_lags, n_bins):
        row = []
        
        # 自回归项（target自身历史）
        for lag in range(1, max_lags+1):
            row.append(target_ts[i-lag])
        
        # 交叉项（reference历史和未来）
        for lag in range(-max_lags, max_lags+1):
            if 0 <= i-lag < n_bins:
                row.append(ref_ts[i-lag])
            else:
                row.append(0)
        
        X.append(row)
    
    X = np.array(X)
    
    # 添加常数项
    X = np.column_stack([np.ones(len(X)), X])
    
    # 拟合Poisson GLM
    try:
        model = GLM(y, X, family=families.Poisson())
        results = model.fit()
        
        # 提取关键参数
        # 交叉项系数（对应lag的index需要调整）
        n_coef = len(results.params)
        
        return {
            'n_coef': n_coef,
            'params': results.params,
            'pvalues': results.pvalues,
            'aic': results.aic,
            'bic': results.bic,
            'converged': results.mle_retvals['converged']
        }
    except Exception as e:
        print(f"  GLM拟合失败: {e}")
        return None


# ===================== 主分析函数 =====================
def run_wodeyar_analysis(cortex_ts, thal_ts, sfreq, tag="run"):
    """
    完整的Wodeyar et al. (2024)分析流程
    
    三种互补方法：
    1. AER (Average Evoked Response)
    2. CCH (Cross-Correlation Histogram)
    3. GLM (Point Process Model)
    """
    print(f"\n{'='*60}")
    print(f"Wodeyar 2024方法分析: {tag}")
    print(f"{'='*60}")
    
    # 1. IED检测（文献方法）
    print("\n[1] IED检测（Wodeyar方法）")
    print(f"  频带: {IED_BAND[0]}-{IED_BAND[1]} Hz")
    print(f"  阈值: {THRESHOLD_MEAN_MULT}×mean")
    print(f"  Fano factor: >{FANO_FACTOR_THRESH}")
    
    cortex_spikes = detect_ied_wodeyar(cortex_ts, sfreq)
    thal_spikes = detect_ied_wodeyar(thal_ts, sfreq)
    
    print(f"\n  皮层spikes: {len(cortex_spikes)}")
    print(f"  丘脑spikes: {len(thal_spikes)}")
    
    if len(cortex_spikes) < 5 or len(thal_spikes) < 5:
        print(f"  ⚠️ Spike数量不足，跳过分析")
        return None
    
    # 2. AER分析
    print("\n[2] AER分析（±1s窗口）")
    times, aer_mean, aer_sem = compute_aer_wodeyar(cortex_spikes, thal_ts, sfreq)
    
    if aer_mean is not None:
        print(f"  AER计算完成，n={len(cortex_spikes)}")
    
    # 3. CCH分析
    print("\n[3] CCH分析（±1s, ~35ms bins, 5000次shuffle）")
    lags, ccg, ci_low, ci_high, ci_low_bonf, ci_high_bonf = compute_ccg_wodeyar(
        cortex_spikes / sfreq, thal_spikes / sfreq
    )
    
    # 计算传播统计
    post_mask = (lags > 0) & (lags <= CCH_WINDOW_SEC * 1000)
    n_spikes_post = np.sum((ccg - ci_high_bonf)[post_mask] * ((ccg - ci_high_bonf)[post_mask] > 0))
    
    print(f"  CCG计算完成")
    print(f"  0-1s窗口内显著丘脑spike数: {n_spikes_post}")
    
    # 4. GLM建模
    print("\n[4] GLM Point Process建模（16ms bins）")
    glm_results = fit_glm_point_process(
        cortex_spikes, thal_spikes, len(cortex_ts), sfreq
    )
    
    if glm_results:
        print(f"  GLM拟合完成")
        print(f"  AIC: {glm_results['aic']:.2f}")
        print(f"  BIC: {glm_results['bic']:.2f}")
        print(f"  收敛: {glm_results['converged']}")
    
    # 5. 绘图
    fig, axes = plt.subplots(3, 1, figsize=(10, 12))
    
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
        axes[0].set_title(f'AER: Cortex→Thalamus (n={len(cortex_spikes)} spikes)')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
    
    # CCH
    axes[1].bar(lags, ccg, width=CCH_BIN_SEC*1000, color='steelblue', alpha=0.6, label='CCG')
    axes[1].fill_between(lags, ci_low_bonf, ci_high_bonf, color='gray', alpha=0.3, label='95% CI (Bonferroni)')
    axes[1].axvline(0, color='red', linestyle='--', alpha=0.5)
    axes[1].set_ylabel('Thalamic Spike Count')
    axes[1].set_title('Cross-Correlation: Cortex→Thalamus')
    axes[1].set_xlabel('Lag (ms)')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    # GLM结果（如果有）
    if glm_results and 'params' in glm_results:
        params = glm_results['params']
        pvals = glm_results['pvalues']
        
        # 显示系数
        x = np.arange(len(params))
        axes[2].bar(x, params, color='steelblue', alpha=0.6)
        axes[2].axhline(0, color='black', linewidth=0.5)
        axes[2].set_ylabel('Coefficient')
        axes[2].set_title('GLM Coefficients (Poisson)')
        axes[2].set_xlabel('Parameter Index')
        axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'/data/shared_home/tlm/Project/MEG-C/results_Wodeyar/{tag}_wodeyar_analysis.png', 
                dpi=150, bbox_inches='tight')
    plt.close()
    
    # 6. 返回结果
    results = {
        'tag': tag,
        'n_cortical_spikes': len(cortex_spikes),
        'n_thalamic_spikes': len(thal_spikes),
        'aer_mean': aer_mean,
        'aer_sem': aer_sem,
        'aer_times': times,
        'ccg_lags': lags,
        'ccg': ccg,
        'ccg_ci_low': ci_low_bonf,
        'ccg_ci_high': ci_high_bonf,
        'n_spikes_post': n_spikes_post,
        'glm_results': glm_results
    }
    
    print(f"\n{'='*60}")
    print(f"分析完成: {tag}")
    print(f"{'='*60}")
    
    return results


if __name__ == "__main__":
    print("Wodeyar et al. (2024) 方法实现")
    print("导入并调用 run_wodeyar_analysis() 函数")
