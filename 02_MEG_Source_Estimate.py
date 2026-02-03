import os
import warnings
import mne
import numpy as np
from scipy.signal import detrend

# 忽略无关警告（与前序代码保持一致）
warnings.filterwarnings("ignore", category=RuntimeWarning)
mne.set_log_level("error")  # 抑制MNE内部非关键警告

# ===================== 全局配置参数（与前序代码完全对齐） =====================
# 1. 基础路径（无需重复设置FreeSurfer，前序代码已配置）
subjects = [f"sub-{i:02d}" for i in range(1, 9)]  # sub-01 ~ sub-08
# subjects = ["sub-01"]  # 测试时可单独指定被试
subjects_dir = "/data/shared_home/tlm/data/MEG-C/freesurfer/mri"
meg_root = "/data/shared_home/tlm/data/MEG-C/spikes6"  # MEG数据根目录
states = ["EC", "EO"]  # 实验状态（闭眼/睁眼）
n_jobs = 4  # 并行计算核心数

# 2. 空房间噪声协方差路径（确认实际路径后修改！）
empty_cov_fname = os.path.join(
    meg_root,
    "empty_room",
    "empty_room",
    "empty_room_empty_room_noise_cov.fif"
)

source_dir = "/data/shared_home/tlm/Project/MEG-C/source"
os.makedirs(source_dir, exist_ok=True)

# ===================== 工具函数（复用前序代码逻辑） =====================
# 工具函数1：模糊匹配MEG文件（解决with_spikes后缀问题）
def find_meg_file(run_path, subject, run, state):
    for fname in os.listdir(run_path):
        if (fname.endswith(".fif") and subject in fname and run in fname and state in fname):
            return os.path.join(run_path, fname)
    return None


# 工具函数2：检查文件/目录是否存在
def check_path(path, path_type="file"):
    if path_type == "file":
        exists = os.path.isfile(path)
    elif path_type == "dir":
        exists = os.path.isdir(path)
    else:
        exists = False
    return exists


# ===================== 批量逆解计算主逻辑 =====================
# 先加载并正则化空房间噪声协方差（只需加载1次，复用所有被试）
if not check_path(empty_cov_fname):
    raise FileNotFoundError(f"空房间噪声协方差文件不存在：{empty_cov_fname}")
noise_cov = mne.read_cov(empty_cov_fname)
print(f"✅ 加载空房间噪声协方差：{empty_cov_fname}")

for subject in subjects:
    print("\n" + "=" * 50)
    print(f"=== 开始处理被试：{subject} ===")
    print("=" * 50)

    # 定义当前被试的BEM目录（保存源空间/前向/STC文件）
    bem_dir = os.path.join(subjects_dir, subject, "bem")
    if not check_path(bem_dir, "dir"):
        print(f"❌ {subject} 的BEM目录不存在 → 跳过该被试")
        continue

    # 1. 加载混合源空间（所有run/状态复用）
    mixed_src_fname = os.path.join(bem_dir, f"{subject}-mixed-src.fif")
    if not check_path(mixed_src_fname):
        print(f"❌ {subject} 混合源空间文件缺失 → 跳过该被试")
        continue
    mixed_src = mne.read_source_spaces(mixed_src_fname)
    print(f"✅ 加载{subject}混合源空间：{mixed_src_fname}")

    # 2. 遍历该被试的所有run目录
    subj_meg_dir = os.path.join(meg_root, subject)
    if not check_path(subj_meg_dir, "dir"):
        print(f"❌ {subject} 无MEG数据目录 → 跳过run处理")
        continue

    # 筛选run目录（仅处理run-开头的目录）
    run_dirs = [d for d in os.listdir(subj_meg_dir)
                if check_path(os.path.join(subj_meg_dir, d), "dir") and d.startswith("run-")]
    if not run_dirs:
        print(f"❌ {subject} 无有效run目录 → 跳过run处理")
        continue

    for run in sorted(run_dirs):
        print(f"\n--- 处理Run：{run} ---")
        run_path = os.path.join(subj_meg_dir, run)

        for state in states:
            print(f"\n  ▶ 处理状态：{state}")
            try:
                # 3. 模糊匹配当前run+state的MEG数据文件
                raw_fname = find_meg_file(run_path, subject, run, state)
                if not raw_fname or not check_path(raw_fname):
                    print(f"  ❌ {state}状态MEG文件未找到 → 跳过")
                    continue
                print(f"  ✅ 找到MEG文件：{os.path.basename(raw_fname)}")

                # 4. 加载MEG数据（仅选MEG通道，不预加载可能报错，故preload=True）
                raw = mne.io.read_raw_fif(raw_fname, preload=True, verbose=False)
                raw.pick(picks="meg")
                info = raw.info

                # 5. 正则化噪声协方差（适配当前MEG数据的info）
                curr_noise_cov = mne.cov.regularize(
                    noise_cov.copy(),  # 复制避免修改原协方差
                    info,
                    mag=0.1,
                    grad=0.1,
                    rank="info"  # 兼容旧版本MNE
                )

                # 6. 加载当前run+state的前向模型
                fwd_fname = os.path.join(bem_dir, f"{subject}-{run}-{state}-mixed-fwd.fif")
                if not check_path(fwd_fname):
                    print(f"  ❌ {state}状态前向模型文件缺失 → 跳过")
                    continue
                fwd = mne.read_forward_solution(fwd_fname, verbose=False)

                # 7. 构建逆算子（核心参数与原代码一致）
                inverse_op = mne.minimum_norm.make_inverse_operator(
                    info,
                    fwd,
                    curr_noise_cov,
                    loose={"surface": 0.2, "volume": 1},  # 分类型设置loose，兼容体积源
                    depth=0.8,
                    rank="info",  # 旧版本兼容参数
                    verbose=False
                )

                # 8. 应用逆解生成源估计（STC）
                stc = mne.minimum_norm.apply_inverse_raw(
                    raw,
                    inverse_op,
                    lambda2=1 / 9.,  # SNR=3，MEG标准值
                    method="dSPM",  # 可替换为sLORETA/mne，根据需求调整
                    verbose=False
                )

                # 9. 保存STC文件（H5格式，兼容后续分析）
                stc_fname = os.path.join(source_dir, subject, run, f"{subject}-{run}-{state}-mixed-stc.h5")
                os.makedirs(os.path.dirname(stc_fname), exist_ok=True)
                stc.save(stc_fname, overwrite=True)
                print(">>> 将要保存的目录：", os.path.dirname(stc_fname))
                print(f"  ✅ STC文件保存成功：{os.path.basename(stc_fname)}")

                # 释放内存（处理大量被试时避免内存溢出）
                del raw, inverse_op, stc, fwd, curr_noise_cov

            except Exception as e:
                print(f"  ❌ {state}状态处理失败：{str(e)[:100]} → 跳过")
                continue

    print(f"\n=== {subject} 所有run处理完成 ===")

print("\n" + "=" * 50)
print("✅ 所有被试逆解计算完成！")
print("=" * 50)