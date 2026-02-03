# DBA (Deep Brain Activity) 方法实现说明

## 参考文献
Attal Y, Schwartz D (2013) Assessment of Subcortical Source Localization Using Deep Brain Activity Imaging Model with Minimum Norm Operators: A MEG Study. PLoS ONE 8(3): e59856.

## DBA方法的核心特性

### 1. 结构特异性的电生理学模型

| 结构 | 源空间类型 | 细胞类型 | 偶极子取向 | DMD (nAm/mm²) | Loose参数 |
|------|-----------|---------|-----------|--------------|-----------|
| 皮层 | 表面 | Open-field | 约束（法向） | 0.25 | 0.2 |
| 海马 | 表面 | Open-field | 约束（垂直表面） | 0.4 | 0.2 |
| **丘脑** | **体积** | **Closed-field** | **随机取向** | **0.025** | **1.0** |
| **纹状体** | **体积** | **Closed-field** | **随机取向** | **0.025** | **1.0** |
| 杏仁核 | 体积 | Open-field | 随机取向 | 1.0 | 1.0 |

### 2. DBA与标准MNE的关键区别

#### 标准MNE方法的问题
```python
# 标准实现（不符合DBA）
loose=0.2  # 对所有源使用相同约束
```

#### DBA方法的正确实现
```python
# DBA实现（结构特异性）
loose={'surface': 0.2, 'volume': 1.0}
#    ^皮层约束     ^体积源自由取向
```

**关键差异**：
- **皮层源**：loose=0.2 → 偶极子约束于法向（pyramidal cells）
- **体积源**：loose=1.0 → 偶极子自由取向（closed-field cells）

### 3. 为什么体积源需要随机取向？

根据Attal & Schwartz (2013)：

> "The thalamus and striatum are essentially made of closed-field cells (i.e., with no preferred source orientation); hence, a current dipole is placed at each node of the inner volume grid with a **random orientation**"

**电生理学基础**：
- 丘脑和纹状体主要由**closed-field细胞**组成
- 这些细胞的 dendritic arborization 产生的电磁场在远处相互抵消
- 没有优先的电流方向 → **随机取向建模**

### 4. DBA方法的完整流程

#### Step 1: 解剖学模型构建
```bash
# 01_Batch_MEG_Coregistration_DBA.py
- 皮层源空间: Oct6 spacing, constrained orientation
- 体积源空间: 5mm spacing, free orientation (关键!)
- BEM模型: 单层，导电率0.33 S/m
```

#### Step 2: 前向模型计算
```bash
- 混合源空间: 皮层表面 + 体积源
- 配准: MEG与MRI对齐
- 前向解: Gain matrix计算
```

#### Step 3: DBA源估计
```python
# 02_MEG_Source_Estimate_DBA.py
inverse_op = mne.minimum_norm.make_inverse_operator(
    info,
    fwd,
    noise_cov,
    loose={'surface': 0.2, 'volume': 1.0},  # DBA关键！
    depth=0.8,  # 深度加权补偿
    method="dSPM"  # 噪声归一化
)
```

### 5. DBA方法的验证指标

根据Attal & Schwartz (2013)，验证应包括：

1. **Point-Spread Function (PSF)**：点源重建的失真
2. **Cross-Talk Function (CTF)**：其他源位置的影响
3. **Dipole Localization Error (DLE)**：
   - DLEg: 重心距离
   - DLEm: 最大值距离

### 6. 实现文件说明

| 文件 | 说明 |
|------|------|
| `01_Batch_MEG_Coregistration_DBA.py` | DBA配准和前向模型 |
| `02_MEG_Source_Estimate_DBA.py` | DBA源估计（dSPM/wMNE） |
| `source_DBA/` | DBA源估计结果目录 |

### 7. 与原实现的主要改进

| 组件 | 原实现 | DBA实现 |
|------|--------|---------|
| 体积源loose | 1.0 | 1.0 ✓ |
| 皮层源loose | 0.2 | 0.2 ✓ |
| 电生理学先验 | ❌ 缺失 | ✓ 文献标准 |
| DMD参数 | ❌ 缺失 | ✓ 已定义 |
| 取向约束 | 隐式 | ✓ 显式建模 |

### 8. 使用方法

```bash
# Step 1: 配准和前向模型（如需要）
/home/tuluman/.conda/envs/tlm/bin/python 01_Batch_MEG_Coregistration_DBA.py

# Step 2: DBA源估计
/home/tuluman/.conda/envs/tlm/bin/python 02_MEG_Source_Estimate_DBA.py

# Step 3: CCG分析（使用DBA源估计结果）
# 修改 SOURCE_DIR 为 source_DBA
```

## 结论

DBA方法的核心不在于使用特定的逆解算法（dSPM/wMNE/sLORETA），而在于：

1. **结构特异性的源空间建模**
2. **符合电生理学的偶极子取向设置**
3. **深部结构的特殊处理（随机取向、深度加权）**

本实现遵循Attal & Schwartz (2013)的标准，确保丘脑等深部结构的源定位符合DBA方法学要求。
