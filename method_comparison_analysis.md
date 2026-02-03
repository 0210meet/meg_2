# Wodeyar et al. (2024) 文献方法对照分析

## 文献核心结论

**"sleep-activated cortical epileptic spikes propagate to the thalamus"**

皮层癫痫棘波**传播至丘脑**（皮层→丘脑方向）

---

## 文献中的关键方法

### 1. IED检测方法

```python
# Wodeyar et al. (2024) 方法
频带: 25-80 Hz (FIR filter)
包络: Hilbert transform
阈值: 3倍平均幅度
规则性: Fano factor > 2.5
合并: 20ms内的spikes合并
```

### 2. 传播分析方法

文献使用了**三种互补方法**：

#### A. AER (Average Evoked Response)
```python
# 皮层棘波触发，观察丘脑响应
window = ±1s around cortical spike
计算平均丘脑响应
检验：与零重叠（Bonferroni校正）
```

#### B. Cross-Correlation Histogram (CCH)
```python
# 未归一化的交叉相关直方图
bins = ~35ms
window = ±1s
显著性：Shuffle检验（5000次）
CI: 95%
多重比较校正：Bonferroni
```

#### C. Point Process Model (GLM)
```python
# 条件强度函数
log λ(t|Ht) = β0 + Σγ·ΔN_cortical + Σβ·ΔN_thalamic
# 广义线性模型
# Poisson分布 + log-link
```

---

## 当前脚本对比

### SEEG_Cortex_Thalamus_CCG_Analysis.py

| 特性 | 文献方法 | 当前实现 | 符合度 |
|------|----------|----------|--------|
| 频带 | 25-80 Hz | 20-80 Hz | ⚠️ 接近 |
| 滤波 | FIR | FIR | ✅ |
| 包络检测 | Hilbert | Hilbert | ✅ |
| Z阈值 | 3×mean | 3.5×std | ⚠️ 不同 |
| Fano factor | >2.5 | ❌ 无 | ❌ 缺失 |
| 形态学约束 | ❌ 无 | ✅ 有 | ❌ 不符合 |
| AER | ✅ ±1s | ✅ ±200ms | ⚠️ 窗口不同 |
| CCH | ✅ 未归一化 | ✅ | ✅ |
| Shuffle | 5000次 | 1000次 | ⚠️ 次数少 |
| GLM建模 | ✅ | ❌ 无 | ❌ 缺失 |

**问题**：
- 使用了额外的形态学约束（文献没有）
- 缺少Fano factor检验
- 缺少Point process建模

### run_thal_cortex_spike_ccg_strong.py

| 特性 | 文献方法 | 当前实现 | 符合度 |
|------|----------|----------|--------|
| 频带 | 25-80 Hz | 20-80 Hz | ⚠️ 接近 |
| Z阈值 | 3×mean | 3.5×std | ⚠️ 不同 |
| 形态学约束 | ❌ 无 | ✅ 有 | ❌ 不符合 |
| CCH | ✅ 未归一化 | ✅ | ✅ |
| Shuffle | 5000次 | 1000次 | ⚠️ 次数少 |
| AER | ❌ | ❌ | ❌ 缺失 |
| GLM建模 | ✅ | ❌ | ❌ 缺失 |

**问题**：
- 添加了文献没有的形态学约束
- 缺少AER分析
- 缺少GLM建模

---

## 关键差异总结

### 文献方法特点

1. **更简单、更保守的IED检测**
   - 没有复杂的形态学约束
   - 使用Fano factor评估振荡规律性
   - 合并20ms内的spikes

2. **互补的三重验证**
   - AER: 时间锁定的平均响应
   - CCH: 时序相关性
   - GLM: 条件强度建模

3. **更严格的统计**
   - 5000次shuffle
   - Bonferroni校正
   - Point process建模

### 当前实现的问题

1. **过度的形态学约束**
   - 添加了文献中没有的FWHM、上升时间等约束
   - 可能导致漏检部分真实的IED

2. **缺少关键分析**
   - 没有Point process建模
   - 没有条件强度函数估计

3. **统计检验不够严格**
   - 仅1000次shuffle（文献5000次）
   - 缺少多重比较校正

---

## 建议的改进方案

### 创建符合文献的方法

需要实现：

1. **简化的IED检测**
```python
def detect_ied_wodeyar(sig, sfreq):
    # 25-80Hz FIR滤波
    sig_filt = filter_data(sig, sfreq, 25, 80, method='fir')
    
    # Hilbert包络
    env = np.abs(hilbert(sig_filt))
    
    # 阈值检测
    thresh = 3 * np.mean(env)
    candidate_peaks = find_peaks(env, height=thresh)[0]
    
    # Fano factor检验
    for cp in candidate_peaks:
        segment = sig[cp-int(0.25*sfreq):cp+int(0.25*sfreq)]
        ff = fano_factor(segment)  # 方差/均值比
        if ff > 2.5:
            keep.append(cp)
    
    # 合并20ms内的spikes
    return merge_spikes(kept, min_interval=20)
```

2. **AER分析**
```python
def compute_aer(cortical_spikes, thalamic_sig, sfreq):
    epochs = []
    for spike_time in cortical_spikes:
        epoch = thalamic_sig[spike_time-sfreq:spike_time+sfreq]  # ±1s
        epochs.append(epoch)
    return np.mean(epochs, axis=0)
```

3. **Cross-correlation**
```python
def compute_ccg_wodeyar(cortical_spikes, thalamic_spikes):
    bins = np.arange(-1, 1+0.035, 0.035)  # ~35ms bins
    hist, _ = np.histogram(thalamic_spikes[:, None] - cortical_spikes, bins=bins)
    
    # 5000次shuffle
    null_dist = []
    for _ in range(5000):
        shuf_spikes = shuffle_intervals(cortical_spikes)
        h, _ = np.histogram(thalamic_spikes[:, None] - shuf_spikes, bins=bins)
        null_dist.append(h)
    
    ci_low = np.percentile(null_dist, 2.5, axis=0)
    ci_high = np.percentile(null_dist, 97.5, axis=0)
    
    # Bonferroni校正
    alpha_corrected = 0.05 / len(bins)
    
    return hist, ci_low, ci_high
```

4. **Point Process Model**
```python
def fit_point_process_model(cortical_spikes, thalamic_spikes, sfreq):
    # 16ms bins (文献使用)
    bin_size = int(0.016 * sfreq)
    
    # 创建二值时间序列
    n_bins = int(len(signal) / bin_size)
    cortical_ts = binarize_spikes(cortical_spikes, n_bins, bin_size)
    thalamic_ts = binarize_spikes(thalamic_spikes, n_bins, bin_size)
    
    # GLM模型
    # log λ(t) = β0 + Σγ·ΔN_cortical(t-i) + Σβ·ΔN_thalamic(t-i)
    
    from statsmodels.api import GLM, families
    X = create_design_matrix(cortical_ts, thalamic_ts, lags=20)
    model = GLM(thalamic_ts, X, family=families.Poisson())
    results = model.fit()
    
    return results
```

---

## 结论

### 当前脚本与文献的主要区别

1. **过度的形态学约束** → 导致IED检出可能不完整
2. **缺少Point process建模** → 无法量化条件强度
3. **统计检验不够严格** → 假阳性率可能更高

### 符合文献的改进方向

1. **简化IED检测**：移除额外的形态学约束
2. **添加Fano factor**：评估振荡规律性
3. **实现GLM建模**：估计条件强度函数
4. **增加shuffle次数**：5000次
5. **多重比较校正**：Bonferroni

### 分析结果差异的可能原因

我们当前分析显示60%丘脑主导，而文献结论是皮层→丘脑传播。可能的原因：

1. **IED检测差异**：形态学约束可能筛选掉了部分皮层棘波
2. **分析窗口差异**：200ms vs 1000ms
3. **统计方法差异**：缺少GLM建模
4. **数据模态差异**：MEG vs SEEG
5. **患者群体差异**：EE-SWAS vs 一般癫痫
