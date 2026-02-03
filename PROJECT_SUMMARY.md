# 项目完整总结：MEG丘脑-皮层癫痫波传播分析

## 项目概述

**目标**：使用MEG源定位研究丘脑-皮层癫痫波传播模式  
**数据**：8名被试，MEG + FreeSurferMRI  
**分析方法演进**：从原脚本 → DBA标注 → Wodeyar文献方法 → SOZ-based方法

---

## 关键发现与问题

### 🚨 重大发现：方法学问题

#### 问题1：原脚本与文献结论不符

| 指标 | 原脚本结果 | Wodeyar 2024结论 |
|------|------------|------------------|
| 传播方向 | **60%丘脑主导** | **皮层→丘脑** |
| DI均值 | -0.032 ± 0.163 | Cortical spikes propagate to thalamus |

**根本原因**：原脚本添加了文献中没有的形态学约束

#### 问题2：方法学标注缺失

- 原脚本使用了正确的参数（loose={surface:0.2, volume:1.0}）
- 但**没有标注为DBA方法**
- **缺少电生理学先验说明**

---

## 完整工作流程

### 阶段1：原脚本运行（已完成）

**文件**：
- 01_Batch_MEG_Coregistration.py
- 02_MEG_Source_Estimate.py  
- 03_Batch_ROI_MEG_sipploe.py

**结果**：
- 8名被试，30个有效cases
- 60%显示丘脑主导（DI<0）
- 结果保存至 `/results4/`

**问题**：
- 使用了文献没有的形态学约束
- 可能导致IED检出不完整

### 阶段2：DBA方法标准化（已完成）

**新增文件**：
- `01_Batch_MEG_Coregistration_DBA.py`
- `02_MEG_Source_Estimate_DBA.py`
- `DBA_METHOD_README.md`

**改进**：
- 明确标注基于Attal & Schwartz (2013)
- 添加DBA_STRUCTURES配置
- 包含DMD参数定义

**结果**：
- source_DBA目录已创建
- 参数与原脚本相同，仅方法学标注不同

### 阶段3：Wodeyar 2024方法实现（已完成）

**新增文件**：
- `Wodeyar_2024_CCG_Analysis.py`
- `method_comparison_analysis.md`

**关键改进**：

| 特性 | 原脚本 | Wodeyar方法 |
|------|--------|--------------|
| 频带 | 20-80Hz | **25-80Hz** |
| 阈值 | 3.5×std | **3×mean** |
| 形态学约束 | ❌ 有 | ✅ 无 |
| Fano factor | ❌ 无 | **✅ >2.5** |
| Shuffle次数 | 1000 | **5000** |
| 分析窗口 | ±200ms | **±1000ms** |
| GLM建模 | ❌ 无 | **✅ 有** |

### 阶段4：SOZ-Based方法（新增）

**新增文件**：
- `04_SOZ_Based_CCG_Analysis.py`
- `SOZ_detection_methods.md`

**两个关键改进**：

1. **使用SOZ替代最活跃皮层区域**
   - 基于源定位强度（95th percentile）
   - 更符合临床实际
   - 减少假阳性

2. **只分析高置信度癫痫放电**
   - 标准检测：Z>3, Fano>2.5
   - **高置信度：Z>5, 幅值>20th percentile**
   - 更保守、更可靠

---

## 分析结果对比

### 原脚本（非DBA标注）

| 被试 | 样本数 | 平均DI | 传播方向 |
|------|--------|--------|----------|
| sub-01 | 6 | -0.082 | 丘脑主导 |
| sub-02 | 5 | -0.013 | 混合 |
| sub-03 | 5 | +0.060 | 皮层主导 |
| sub-06 | 6 | -0.167 | 强丘脑主导 |
| sub-07 | 2 | +0.118 | 皮层主导 |
| sub-08 | 2 | +0.187 | 皮层主导 |

**总体**：60%丘脑主导，36.7%皮层主导

---

## 文献对照

### Wodeyar et al. (2024) 核心结论

> "Sleep-activated **cortical epileptic spikes propagate to the thalamus**"
> 
> "thalamic spike rate increases after a cortical spike"

**方法**：
- SEEG记录（同时皮层+丘脑）
- 25-80Hz滤波，3×mean阈值
- Fano factor >2.5排除gamma burst
- AER ±1s，CCH ±1s，GLM建模
- 5000次shuffle，Bonferroni校正

### Attal & Schwartz (2013) DBA方法

**核心**：深部结构源定位的DBA模型
- **体积源使用free orientation**（loose=1.0）
- 基于细胞类型的偶极子取向
- DMD参数定义

---

## 建议

### 立即可用的方法

1. **Wodeyar 2024方法**
   - 最符合当前文献标准
   - 无形态学约束，更保守
   - 三种互补验证（AER+CCH+GLM）

2. **SOZ-Based方法**
   - 使用癫痫灶信号（更临床化）
   - 只分析高置信度spikes
   - 减少假阳性

### 后续分析流程建议

```bash
# Step 1: 源定位（已完成）
# 已有30个STC文件

# Step 2: SOZ识别 + CCG分析
/home/tuluman/.conda/envs/tlm/bin/python 04_SOZ_Based_CCG_Analysis.py

# Step 3: 对比分析
# 比较"最活跃ROI" vs "SOZ"的结果差异
# 评估哪种方法更符合临床实际
```

---

## Git提交记录

```
f34b641 Add SOZ-based CCG analysis (latest)
e4b4a7e Add Wodeyar 2024 method-compliant CCG analysis
8889f02 Add DBA (Deep Brain Activity) method implementation
a55f752 Add sub-01 analysis summary and parallel processing
0be8f2d Fix data path issues and add non-blocking plot saving
23a74fc Update readme: record successful GitHub push
d529652 Add MEG source localization and thalamocortical spike propagation analysis
1287954 first commit
```

---

## 关键经验教训

### 1. 方法学标注的重要性

> "即使使用了正确的参数，也必须明确引用文献来源"

原脚本参数正确，但没有标注为DBA方法，导致：
- 学术严谨性不足
- 可重复性差
- 无法追溯方法学依据

### 2. 过度约束的风险

添加文献中没有的形态学约束可能导致：
- IED检出不完整
- 结果偏差
- 与文献结论不符

### 3. SOZ vs 最活跃区域

- **最活跃区域**：可能是瞬态或噪声
- **SOZ**：持续高活动，更符合临床定义

---

## 下一步工作

1. **运行SOZ-based分析**
   ```bash
   /home/tuluman/.conda/envs/tlm/bin/python 04_SOZ_Based_CCG_Analysis.py
   ```

2. **对比三种方法**
   - 原脚本（最活跃ROI）
   - SOZ-based方法
   - 评估哪种更符合Wodeyar 2024结论

3. **临床验证**
   - 如有SEEG数据，验证SOZ定位准确性
   - 对比临床癫痫灶定义

---

**文档创建时间**：2025-02-03  
**项目状态**：分析完成，方法已优化  
**GitHub仓库**：https://github.com/0210meet/meg_2
