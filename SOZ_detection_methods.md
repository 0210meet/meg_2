# 癫痫灶（SOZ）检测方法

## 方法1: 基于源定位强度的SOZ识别

```python
def identify_soz_from_stc(stc, percentile_thresh=95):
    """
    基于STC结果识别癫痫灶
    
    原理：癫痫灶通常显示持续的、高幅值的异常活动
    """
    # 1. 计算每个源点的平均激活强度
    source_power = np.mean(np.abs(stc.data), axis=1)
    
    # 2. 设定阈值（如95th percentile）
    threshold = np.percentile(source_power, percentile_thresh)
    
    # 3. 识别超过阈值的源点
    soz_sources = np.where(source_power > threshold)[0]
    
    # 4. 聚类相邻的源点形成SOZ区域
    soz_labels = cluster_sources(stc.vertices_or VERtno, soz_sources)
    
    return soz_labels, soz_sources
```

## 方法2: 基于IED密度的SOZ识别

```python
def identify_soz_from_ied_density(ied_times, stc, time_window=0.5):
    """
    基于IED出现频率识别癫痫灶
    
    原理：癫痫灶通常IED密度最高
    """
    # 1. 统计每个源点在IED时间点的激活
    ied_activated = np.zeros(stc.data.shape[0])
    
    for ied_time in ied_times:
        time_idx = np.argmin(np.abs(stc.times - ied_time))
        ied_activated += np.abs(stc.data[:, time_idx])
    
    # 2. 归一化
    ied_density = ied_activated / len(ied_times)
    
    # 3. 识别高密度区域
    threshold = np.percentile(ied_density, 90)
    soz_sources = np.where(ied_density > threshold)[0]
    
    return soz_sources, ied_density
```

## 方法3: 临床定义的SOZ

如果有SEEG或临床定义的SOZ：

```python
# 使用临床标记的SOZ标签
CLINICAL_SOZ = {
    'sub-01': ['transversetemporal-lh'],  # 示例
    'sub-02': ['precentral-lh'],
    # ...
}

def get_clinical_soz(subject, available_labels):
    """获取临床定义的SOZ"""
    if subject in CLINICAL_SOZ:
        for soz_label in CLINICAL_SOZ[subject]:
            for label in available_labels:
                if soz_label in label.name:
                    return label
    return None
```

## 实际应用建议

### 结合多种方法

```python
def identify_soz_multi_method(stc, ied_times, subject):
    """
    结合多种方法识别SOZ
    """
    # 方法1: 源激活强度
    soz_by_power, _ = identify_soz_from_stc(stc, percentile_thresh=95)
    
    # 方法2: IED密度
    soz_by_density, _ = identify_soz_from_ied_density(ied_times, stc)
    
    # 方法3: 临床定义（如果有）
    clinical_soz = get_clinical_soz(subject, labels_cortex)
    
    # 综合判断
    if clinical_soz:
        return clinical_soz  # 优先使用临床定义
    else:
        # 使用高激活+高密度的区域
        overlap = np.intersect1d(soz_by_power, soz_by_density)
        if len(overlap) > 0:
            # 创建SOZ label
            soz_label = create_soz_label(stc.src, overlap)
            return soz_label
        else:
            # 使用高功率区域
            soz_label = create_soz_label(stc.src, soz_by_power)
            return soz_label
```

### 优势

1. **更符合临床实际**：癫痫灶是最活跃的区域
2. **减少假阳性**：避免使用偶发的最活跃区域
3. **可重复性**：基于客观数据而非随机最大值

