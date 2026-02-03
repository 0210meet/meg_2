#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
01_Batch_MEG_Coregistration_DBA.py

基于DBA模型的MEG配准和前向模型构建
参考: Attal & Schwartz (2013) PLoS ONE 8(3): e59856

DBA关键改进：
1. 体积源使用随机取向（free orientation）用于丘脑等closed-field结构
2. 符合电生理学模型的源空间构建
"""

import os
import warnings
import mne
from mne.coreg import Coregistration

warnings.filterwarnings("ignore", category=RuntimeWarning)

# ===================== 基本路径配置 =====================
freesurfer_home = "/data/shared_home/tlm/tool/freesurfer"
os.environ["FREESURFER_HOME"] = freesurfer_home

# 添加FreeSurfer工具目录到PATH
fs_bin_dirs = [
    os.path.join(freesurfer_home, "bin"),
    os.path.join(freesurfer_home, "mri", "bin"),
    os.path.join(freesurfer_home, "tktools"),
]
for dir in fs_bin_dirs:
    os.environ["PATH"] += os.pathsep + dir

# ===================== DBA配置 =====================
# DBA模型的结构特异性参数（参考Attal & Schwartz 2013 Table 1）
DBA_CONFIG = {
    'thalamus': {
        'cell_type': 'closed',      # Closed-field cells
        'orientation': 'free',      # 随机取向（关键！）
        'DMD': 0.025,              # nAm/mm²
        'volume_spacing_mm': 5.0,   # 体积源间距
    },
    'striatum': {
        'cell_type': 'closed',
        'orientation': 'free',
        'DMD': 0.025,
        'volume_spacing_mm': 5.0,
    },
    'amygdala': {
        'cell_type': 'open',
        'orientation': 'free',
        'DMD': 1.0,
        'volume_spacing_mm': 5.0,
    }
}

# ===================== 项目配置 =====================
subjects = [f"sub-{i:02d}" for i in range(1, 9)]
subjects_dir = "/data/shared_home/tlm/data/MEG-C/freesurfer/mri"
meg_root = "/data/shared_home/tlm/data/MEG-C/spikes6"
states = ["EC", "EO"]
n_jobs = 4
ico = 4

# ===================== 工具函数 =====================
def find_meg_file(run_path, subject, run, state):
    """模糊匹配MEG文件"""
    for fname in os.listdir(run_path):
        if (fname.endswith(".fif") and subject in fname and run in fname and state in fname):
            return os.path.join(run_path, fname)
    return None

def check_path(path, path_type="file"):
    """检查文件/目录是否存在"""
    if path_type == "file":
        return os.path.isfile(path)
    elif path_type == "dir":
        return os.path.isdir(path)
    return False

# ===================== 批量处理主逻辑 =====================
for subject in subjects:
    print("\n=== Subject:", subject, "===")
    bem_dir = os.path.join(subjects_dir, subject, "bem")
    os.makedirs(bem_dir, exist_ok=True)

    # === 1. 生成/加载 BEM模型 ===
    fname_bem_model = os.path.join(bem_dir, f"{subject}-5120-bem.fif")
    fname_bem_sol = os.path.join(bem_dir, f"{subject}-5120-bem-sol.fif")

    if not check_path(fname_bem_model):
        try:
            mne.bem.make_watershed_bem(
                subject=subject,
                subjects_dir=subjects_dir,
                overwrite=True,
                verbose=True
            )
            conductivity = (0.33,)
            bem_model = mne.make_bem_model(
                subject=subject,
                ico=ico,
                conductivity=conductivity,
                subjects_dir=subjects_dir,
                verbose=True
            )
            mne.write_bem_surfaces(fname_bem_model, bem_model, overwrite=True)
            print(f"  ✅ 单层BEM几何模型保存: {fname_bem_model}")
        except Exception as e:
            print(f"  ❌ 生成BEM模型失败: {e}")
            continue
    else:
        bem_model = mne.read_bem_surfaces(fname_bem_model)
        print(f"  ✅ 加载已有BEM几何模型")

    if not check_path(fname_bem_sol):
        try:
            bem_sol = mne.make_bem_solution(bem_model, verbose=True)
            mne.write_bem_solution(fname_bem_sol, bem_sol)
            print(f"  ✅ BEM解保存: {fname_bem_sol}")
        except Exception as e:
            print(f"  ❌ 生成BEM解失败: {e}")
            continue
    else:
        bem_sol = mne.read_bem_solution(fname_bem_sol)
        print(f"  ✅ 加载已有BEM解")

    # === 2. 生成头皮表面 ===
    scalp_surf_fname = os.path.join(bem_dir, f"{subject}-head.fif")
    if not check_path(scalp_surf_fname):
        try:
            mne.bem.make_scalp_surfaces(
                subject=subject,
                subjects_dir=subjects_dir,
                force=True,
                overwrite=True,
                verbose=False
            )
            print(f"  ✅ 头皮表面生成")
        except Exception as e:
            print(f"  ❌ 生成头皮表面失败: {e}")
            continue

    # === 3. 构建皮层源空间 ===
    surf_src_fname = os.path.join(bem_dir, f"{subject}-surf-src.fif")
    if not check_path(surf_src_fname):
        surf_src = mne.setup_source_space(
            subject=subject,
            subjects_dir=subjects_dir,
            spacing='oct6',
            add_dist=False,
            verbose=False
        )
        mne.write_source_spaces(surf_src_fname, surf_src, overwrite=True)
        print(f"  ✅ 皮层源空间保存")
    else:
        surf_src = mne.read_source_spaces(surf_src_fname)
        print(f"  ✅ 加载已有皮层源空间")
    print(f"  皮层源空间: {sum(s['nuse'] for s in surf_src)} 顶点")

    # === 4. 构建DBA体积源空间（关键修改！）===
    src_vol_fname = os.path.join(bem_dir, f"{subject}-vol-src-for-DBA.fif")
    if not check_path(src_vol_fname):
        # DBA方法：使用aseg分割定义深部结构
        fname_aseg = os.path.join(subjects_dir, subject, "mri", "aseg.mgz")
        
        # 丘脑标签（双侧）
        labels_vol = ['Left-Thalamus-Proper', 'Right-Thalamus-Proper', 'Thalamus-Proper']
        
        # DBA关键：体积源使用5mm间距，允许随机取向
        vol_src = mne.setup_volume_source_space(
            subject=subject,
            mri=fname_aseg,
            pos=5.0,  # 5mm间距（符合DBA标准）
            bem=fname_bem_model,
            volume_label=labels_vol,
            subjects_dir=subjects_dir,
            add_interpolator=False,  # 提速
            verbose=True,
        )
        mne.write_source_spaces(src_vol_fname, vol_src, overwrite=True)
        print(f"  ✅ DBA体积源空间保存")
        print(f"  ⚠ DBA配置: 体积源将使用free orientation (loose=1.0)")
    else:
        vol_src = mne.read_source_spaces(src_vol_fname)
        print(f"  ✅ 加载已有体积源空间")
    print(f"  体积源空间: {len(vol_src)} 个子空间, {sum(s['nuse'] for s in vol_src)} 顶点")

    # === 5. 合并为混合源空间 ===
    mixed_src_fname = os.path.join(bem_dir, f"{subject}-mixed-src.fif")
    if not check_path(mixed_src_fname):
        mixed_src = surf_src + vol_src
        mne.write_source_spaces(mixed_src_fname, mixed_src, overwrite=True)
        print(f"  ✅ 混合源空间保存")
    else:
        mixed_src = mne.read_source_spaces(mixed_src_fname)
        print(f"  ✅ 加载已有混合源空间")

    # === 6. 对每个run+状态做配准和前向模型 ===
    subj_meg_dir = os.path.join(meg_root, subject)
    if not check_path(subj_meg_dir, "dir"):
        print(f"  ❌ {subject} 无MEG目录")
        continue

    run_dirs = [d for d in os.listdir(subj_meg_dir)
                if check_path(os.path.join(subj_meg_dir, d), "dir") and d.startswith("run-")]
    if not run_dirs:
        print(f"  ❌ {subject} 无有效run目录")
        continue

    for run in sorted(run_dirs):
        run_path = os.path.join(subj_meg_dir, run)
        print(f"\n  --- Run: {run} ---")
        
        trans_fname = os.path.join(bem_dir, f"{subject}-{run}-trans.fif")
        trans = None
        if check_path(trans_fname):
            trans = mne.read_trans(trans_fname)
            print(f"      ✅ 加载已有配准文件")

        for state in states:
            raw_fname = find_meg_file(run_path, subject, run, state)
            if not raw_fname or not check_path(raw_fname):
                print(f"    ❌ {state} MEG文件未找到")
                continue
            print(f"    处理状态: {state}")

            try:
                raw = mne.io.read_raw_fif(raw_fname, preload=False, verbose=False)
                info = raw.info

                if "dig" not in info or len(info["dig"]) < 3:
                    print(f"    ❌ 无数字化定位数据")
                    continue

                # === Coregistration ===
                if trans is None:
                    coreg = Coregistration(info, subject=subject,
                                           subjects_dir=subjects_dir,
                                           fiducials="auto")
                    coreg.fit_fiducials()
                    coreg.set_scale_mode("none")
                    coreg.fit_icp(n_iterations=20, nasion_weight=2.0, verbose=False)
                    mne.write_trans(trans_fname, coreg.trans, overwrite=True)
                    print(f"      ✅ 配准文件保存")
                    trans = coreg.trans

                # === Forward solution ===
                fwd_fname = os.path.join(bem_dir, f"{subject}-{run}-{state}-mixed-fwd.fif")
                if not check_path(fwd_fname):
                    fwd = mne.make_forward_solution(
                        info,
                        trans=trans,
                        src=mixed_src,
                        bem=fname_bem_sol,
                        mindist=3.0,
                        meg=True,
                        eeg=False,
                        n_jobs=n_jobs,
                        verbose=False
                    )
                    mne.write_forward_solution(fwd_fname, fwd, overwrite=True)
                    print(f"      ✅ 前向模型保存")
                else:
                    print(f"      ✅ 加载已有前向模型")

            except Exception as e:
                print(f"    ❌ 处理失败: {e}")
                continue

    print(f"=== {subject} 处理完成 ===\n")

print("✅ 所有被试DBA配准完成!")
print("\nDBA方法特性总结：")
print("  ✓ 体积源间距: 5mm (符合DBA标准)")
print("  ✓ 丘脑标签: Left/Right-Thalamus-Proper")
print("  ✓ 后续源估计将使用 loose={'surface':0.2, 'volume':1.0}")
print("    → 体积源free orientation模拟closed-field细胞")
