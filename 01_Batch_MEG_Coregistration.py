import os
import warnings
import mne
from mne.coreg import Coregistration

# 忽略无关警告（如文件名规范警告）
warnings.filterwarnings("ignore", category=RuntimeWarning)

# === 1. 基本路径 & 参数 (根据你实际情况修改) ===
freesurfer_home = "/data/shared_home/tlm/tool/freesurfer"
os.environ["FREESURFER_HOME"] = freesurfer_home

# 2. 添加 FreeSurfer 所有工具目录到 PATH
fs_bin_dirs = [
    os.path.join(freesurfer_home, "bin"),
    os.path.join(freesurfer_home, "mri", "bin"),
    os.path.join(freesurfer_home, "tktools"),
]
for dir in fs_bin_dirs:
    os.environ["PATH"] += os.pathsep + dir

# --- 配置部分: 根据你的实际项目结构修改 ---
# subjects = [f"sub-{i:02d}" for i in range(1, 9)]  # sub-01 … sub-08
subjects = ["sub-01"]
subjects_dir = "/data/shared_home/tlm/data/MEG-C/freesurfer/mri"
# save_dir = "/data/shared_home/tlm/data/"
meg_root = "/data/shared_home/tlm/data/MEG-C/spikes6"  # MEG 数据根目录
states = ["EC", "EO"]  # 实验状态 (闭眼 / 睁眼)
n_jobs = 4  # 并行计算核心数
ico = 4  # BEM网格分辨率（ico4=5120面片/层，MEG单层足够）


# 工具函数：模糊匹配MEG文件（解决with_spikes后缀问题）
def find_meg_file(run_path, subject, run, state):
    for fname in os.listdir(run_path):
        if (fname.endswith(".fif") and subject in fname and run in fname and state in fname):
            return os.path.join(run_path, fname)
    return None


# 工具函数：检查文件/目录是否存在
def check_path(path, path_type="file"):
    if path_type == "file":
        exists = os.path.isfile(path)
    elif path_type == "dir":
        exists = os.path.isdir(path)
    else:
        exists = False
    return exists


for subject in subjects:
    print("\n=== Subject:", subject, "===")
    bem_dir = os.path.join(subjects_dir, subject, "bem")
    os.makedirs(bem_dir, exist_ok=True)

    # === 1. 生成 / 读取 单层BEM几何模型(bem.fif) + BEM解(bem-sol.fif) ===
    # 定义文件名（单层BEM仍保留5120后缀，与MEG常规命名一致）
    fname_bem_model = os.path.join(bem_dir, f"{subject}-5120-bem.fif")  # 单层BEM几何文件
    fname_bem_sol = os.path.join(bem_dir, f"{subject}-5120-bem-sol.fif")  # 单层BEM解文件

    # 步骤1：生成单层BEM几何模型（仅内颅骨，核心修改！）
    if not check_path(fname_bem_model):
        try:
            # 第一步：用watershed算法分割（即使单层也需要基础表面，仅提取内颅骨）
            mne.bem.make_watershed_bem(
                subject=subject,
                subjects_dir=subjects_dir,
                overwrite=True,
                verbose=True
            )
            # 第二步：构建单层BEM几何模型（仅内颅骨inner_skull）
            # 导电率：单层仅需0.33 S/m（大脑/脑脊液导电率）
            conductivity = (0.33,)  # 核心修改1：单层导电率
            bem_model = mne.make_bem_model(
                subject=subject,
                ico=ico,  # 对应5120分辨率
                conductivity=conductivity,
                subjects_dir=subjects_dir,
                verbose=True
            )
            # 保存单层BEM几何模型为bem.fif
            mne.write_bem_surfaces(fname_bem_model, bem_model, overwrite=True)
            print(f"  ✅ 单层BEM几何模型保存: {fname_bem_model}")
        except Exception as e:
            print(f"  ❌ 生成单层BEM几何模型失败: {e} → 跳过该被试")
            continue
    else:
        # 加载已存在的单层BEM几何模型
        bem_model = mne.read_bem_surfaces(fname_bem_model)
        print(f"  ✅ 加载已有单层BEM几何模型: {fname_bem_model}")

    # 步骤2：生成/加载单层BEM解（bem-sol.fif）
    if not check_path(fname_bem_sol):
        try:
            # 基于单层BEM几何模型计算BEM解
            bem_sol = mne.make_bem_solution(bem_model, verbose=True)
            mne.write_bem_solution(fname_bem_sol, bem_sol)
            print(f"  ✅ 单层BEM解保存: {fname_bem_sol}")
        except Exception as e:
            print(f"  ❌ 生成单层BEM解失败: {e} → 跳过该被试")
            continue
    else:
        bem_sol = mne.read_bem_solution(fname_bem_sol)
        print(f"  ✅ 加载已有单层BEM解: {fname_bem_sol}")

    # === 补充：生成头皮表面（配准依赖，保留原逻辑） ===
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
            print(f"  ✅ 头皮表面生成: {scalp_surf_fname}")
        except Exception as e:
            print(f"  ❌ 生成头皮表面失败: {e} → 跳过该被试")
            continue

    # === 2. 构建 surface-based source space (皮层) ===
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
        print(f"  ✅ 皮层源空间保存: {surf_src_fname}")
    else:
        surf_src = mne.read_source_spaces(surf_src_fname)
        print(f"  ✅ 加载已有皮层源空间")
    print(f"  皮层源空间: {sum(s['nuse'] for s in surf_src)} 顶点（每半球约）")

    # === 3. 构建 volume-based source space (深部结构 + 全脑) ===
    # 关键：体积源空间仍传入单层BEM几何文件（fname_bem_model）
    src_vol_fname = os.path.join(bem_dir, f"{subject}-vol-src-for-mixed.fif")
    if not check_path(src_vol_fname):
        # 可选：定义需要的深部结构标签（丘脑）
        labels_vol = ['Left-Thalamus-Proper', 'Right-Thalamus-Proper', 'Thalamus-medial']

        # fname_aseg = os.path.join(subjects_dir, subject, "mri", "aseg.mgz")
        fname_aseg= os.path.join(subjects_dir, subject, "mri", "aseg.mgz")


        vol_src = mne.setup_volume_source_space(
            subject=subject,
            mri=fname_aseg,  # 基于aseg分割深部结构
            pos=5.0,  # 体积源点间距（mm）
            bem=fname_bem_model,  # 传入单层BEM几何文件
            volume_label=labels_vol,  # 仅保留丘脑（可选）
            subjects_dir=subjects_dir,
            add_interpolator=False,  # 提速，生产环境建议设为True
            verbose=True,
        )
        mne.write_source_spaces(src_vol_fname, vol_src, overwrite=True)
        print(f"  ✅ 体积源空间保存: {src_vol_fname}")
    else:
        vol_src = mne.read_source_spaces(src_vol_fname)
        print(f"  ✅ 加载已有体积源空间")
    print(f"  体积源空间: {len(vol_src)} 个子空间, {sum(s['nuse'] for s in vol_src)} 顶点")

    # === 4. 合并为 mixed source space ===
    mixed_src_fname = os.path.join(bem_dir, f"{subject}-mixed-src.fif")
    if not check_path(mixed_src_fname):
        mixed_src = surf_src + vol_src
        mne.write_source_spaces(mixed_src_fname, mixed_src, overwrite=True)
        print(f"  ✅ 混合源空间保存: {mixed_src_fname}")
    else:
        mixed_src = mne.read_source_spaces(mixed_src_fname)
        print(f"  ✅ 加载已有混合源空间")

    # === 5. 对每个 run + 状态 做 coreg + forward solution ===
    subj_meg_dir = os.path.join(meg_root, subject)
    if not check_path(subj_meg_dir, "dir"):
        print(f"  ❌ {subject} 无MEG目录 → 跳过run处理")
        continue

    # 筛选 run 目录（仅处理 run- 开头的目录）
    run_dirs = [d for d in os.listdir(subj_meg_dir)
                if check_path(os.path.join(subj_meg_dir, d), "dir") and d.startswith("run-")]
    if not run_dirs:
        print(f"  ❌ {subject} 无有效run目录 → 跳过run处理")
        continue

    for run in sorted(run_dirs):
        run_path = os.path.join(subj_meg_dir, run)
        print(f"\n  --- Run: {run} ---")
        # 同一run内EC/EO复用trans.fif（优化点）
        trans_fname = os.path.join(bem_dir, f"{subject}-{run}-trans.fif")
        trans = None
        if check_path(trans_fname):
            trans = mne.read_trans(trans_fname)
            print(f"      ✅ 加载已有run级配准文件: {trans_fname}")

        for state in states:
            # 模糊匹配 MEG 文件
            raw_fname = find_meg_file(run_path, subject, run, state)
            if not raw_fname or not check_path(raw_fname):
                print(f"    ❌ {state} 状态下MEG文件未找到 → 跳过")
                continue
            print(f"    处理状态: {state}")

            try:
                raw = mne.io.read_raw_fif(raw_fname, preload=False, verbose=False)
                info = raw.info

                # 检查数字化信息（配准必需）
                if "dig" not in info or len(info["dig"]) < 3:
                    print(f"    ❌ MEG文件无数字化定位数据 → 跳过")
                    continue

                # === Coregistration (head-shape ↔ MRI) ===
                if trans is None:  # 同一run仅计算1次trans
                    coreg = Coregistration(info, subject=subject,
                                           subjects_dir=subjects_dir,
                                           fiducials="auto")
                    coreg.fit_fiducials()
                    coreg.set_scale_mode("none")
                    coreg.fit_icp(n_iterations=20, nasion_weight=2.0, verbose=False)
                    mne.write_trans(trans_fname, coreg.trans, overwrite=True)
                    print(f"      ✅ Run级配准文件保存: {trans_fname}")
                    trans = coreg.trans

                # === Forward solution (混合源空间) ===
                # 关键：前向模型传入单层BEM解文件
                fwd_fname = os.path.join(bem_dir, f"{subject}-{run}-{state}-mixed-fwd.fif")
                if not check_path(fwd_fname):
                    fwd = mne.make_forward_solution(
                        info,  # 从raw提取info，无需evoked文件
                        trans=trans,
                        src=mixed_src,
                        bem=fname_bem_sol,  # 传入单层BEM解
                        mindist=3.0,  # 忽略距内颅骨≤5mm的源点
                        meg=True,
                        eeg=False,
                        n_jobs=n_jobs,
                        verbose=False
                    )
                    mne.write_forward_solution(fwd_fname, fwd, overwrite=True)
                    print(f"      ✅ 混合源空间前向模型保存: {fwd_fname}")
                else:
                    print(f"      ✅ 加载已有混合源空间前向模型: {fwd_fname}")

            except Exception as e:
                print(f"    ❌ 处理{state}状态失败: {e} → 跳过")
                continue

    print(f"=== {subject} 处理完成 ===\n")

print("✅ 所有被试处理完成!")