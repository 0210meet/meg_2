import os
import mne
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
# from run_thal_cortex_spike_ccg_strong import run_bidirectional_spike_ccg
from SEEG_Cortex_Thalamus_CCG_Analysis import run_spike_propagation_analysis
# from strong import run_single_run_strong
from scipy.signal import correlate
from scipy.stats import zscore
from scipy.stats import bootstrap
from scipy.stats import gaussian_kde

# ===================== 0. 全局配置与前置修复 =====================
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'SimHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

# 1. 批量核心配置
SUBJECTS = [f"sub-{i:02d}" for i in range(1, 9)]  # sub-01 ~ sub-08
# SUBJECTS = ["sub-01"]  # 测试时可单独指定被试
STATES = ["EC", "EO"]  # 实验状态：闭眼/睁眼

# 2. 路径配置（与逆解代码完全对齐）
SOURCE_DIR = "/data/shared_home/tlm/Project/MEG-C/source"  # STC文件保存目录
FREESURFER_DIR = "/data/shared_home/tlm/data/MEG-C/freesurfer/mri"  # Freesurfer解剖数据目录
SAVE_DIR = "/data/shared_home/tlm/Project/MEG-C/results4"  # CCG结果保存目录
os.makedirs(SAVE_DIR, exist_ok=True)

# 3. 丘脑源空间索引映射
THAL_SRC_MAP = {
    "left": {"src_idx": 2, "name": "Left-Thalamus-Proper", "hemi": "lh"},
    "right": {"src_idx": 3, "name": "Right-Thalamus-Proper", "hemi": "rh"}
}


# ===================== 工具函数（整合优化） =====================
def check_path(path, path_type="file"):
    """检查文件/目录是否存在（区分类型）"""
    if path_type == "file":
        return os.path.isfile(path)
    elif path_type == "dir":
        return os.path.isdir(path)
    return False


def find_stc_file(run_path, subject, run, state):
    """模糊匹配STC文件（适配不同命名后缀）"""
    for fname in os.listdir(run_path):
        if fname.endswith(".h5") and all([x in fname for x in [subject, run, state]]):
            return os.path.join(run_path, fname)
    return None


def build_vertno_map(src):
    """构建全局顶点号 → stc.data局部索引的映射表"""
    vertno_map = {}
    local_idx = 0
    for s in src:
        for vert in s["vertno"]:
            vertno_map[vert] = local_idx
            local_idx += 1
    return vertno_map


def find_most_active_dipole(roi_verts_local, stc_data):
    """找到ROI内幅值最大的单个偶极子"""
    if len(roi_verts_local) == 0:
        print("⚠️ ROI无有效偶极子！")
        return -1, 0, np.zeros(stc_data.shape[1])

    roi_ts = stc_data[roi_verts_local, :]
    dipole_activity = np.max(np.abs(roi_ts), axis=1)
    max_idx_in_roi = np.argmax(dipole_activity)
    dipole_idx_local = roi_verts_local[max_idx_in_roi]
    dipole_max_amp = dipole_activity[max_idx_in_roi]
    dipole_ts = stc_data[dipole_idx_local, :]
    return dipole_idx_local, dipole_max_amp, dipole_ts


def process_single_case(subject, run, state, src):
    """
    处理单个被试-单个run-单个状态（复用已加载的src）
    返回：True/False（成功/失败）
    """
    try:
        # 1. 定位STC文件
        run_path = os.path.join(SOURCE_DIR, subject, run)
        stc_fname = find_stc_file(run_path, subject, run, state)
        if not stc_fname or not check_path(stc_fname):
            print(f"❌ {subject}-{run}-{state}：STC文件缺失 → 跳过")
            return False

        # 2. 读取STC数据
        stc = mne.read_source_estimate(stc_fname)
        times = stc.times
        sfreq = 1 / (times[1] - times[0])
        vertno_map = build_vertno_map(src)
        print(f"✅ {subject}-{run}-{state}：数据加载完成 | 采样频率：{sfreq:.1f}Hz")

        # 3. 提取皮层ROI（aparc分区）
        labels_cortex = mne.read_labels_from_annot(subject, parc="aparc", subjects_dir=FREESURFER_DIR)
        label_ts_cortex = mne.extract_label_time_course(
            [stc], labels_cortex, src, mode="mean", allow_empty=True
        )[0]
        label_ts_cortex = np.asarray(label_ts_cortex)

        # 4. 筛选最活跃的皮层ROI
        roi_info_cortex = [{
            'roi_idx': idx,
            'name': label.name,
            'verts_local': [vertno_map[v] for v in label.vertices if v in vertno_map],
            'max_activity': np.abs(label_ts_cortex[idx]).max()
        } for idx, label in enumerate(labels_cortex)]
        roi_info_cortex.sort(key=lambda x: x["max_activity"], reverse=True)
        most_active_cortex = roi_info_cortex[0]
        cortex_side = "left" if "lh" in most_active_cortex["name"] else "right"
        print(f"🏆 {subject}-{run}-{state}：最活跃皮层ROI → {most_active_cortex['name']}（{cortex_side}半球）")

        # 5. 构建同侧丘脑ROI
        thal_config = THAL_SRC_MAP[cortex_side]
        thal_verts_global = src[thal_config["src_idx"]]["vertno"]
        thal_verts_local = [vertno_map[v] for v in thal_verts_global if v in vertno_map]
        if len(thal_verts_local) == 0:
            print(f"❌ {subject}-{run}-{state}：丘脑无有效偶极子 → 跳过")
            return False

        thal_label = mne.Label(vertices=thal_verts_global, hemi=thal_config["hemi"], name=thal_config["name"],
                               subject=subject)
        label_ts_thal = mne.extract_label_time_course([stc], [thal_label], src, mode="mean", allow_empty=True)[0][0]

        # 6. 找最活跃偶极子
        cortex_dipole_idx, cortex_dipole_amp, cortex_dipole_ts = find_most_active_dipole(
            most_active_cortex["verts_local"], stc.data
        )
        thal_dipole_idx, thal_dipole_amp, thal_dipole_ts = find_most_active_dipole(
            thal_verts_local, stc.data
        )
        print(
            f"⚡ {subject}-{run}-{state}：核心偶极子 | 皮层幅值：{cortex_dipole_amp:.2f}nAm | 丘脑幅值：{thal_dipole_amp:.2f}nAm")

        # 7. 运行CCG分析并保存结果
        tag = f"{subject}_{run}_{state}"
        case_save_dir = os.path.join(SAVE_DIR, subject, run, state)
        os.makedirs(case_save_dir, exist_ok=True)
        ccg_results = run_spike_propagation_analysis(cortex_dipole_ts, thal_dipole_ts, sfreq, tag=tag, save_dir=case_save_dir)
        np.save(os.path.join(case_save_dir, f"{tag}_ccg_results.npy"), ccg_results)
        print(f"💾 {subject}-{run}-{state}：结果保存至 {case_save_dir}")

        # 释放内存
        del stc, label_ts_cortex, cortex_dipole_ts, thal_dipole_ts, ccg_results
        return True

    except Exception as e:
        print(f"❌ {subject}-{run}-{state}：处理失败 → {str(e)[:100]}")
        return False


def get_total_cases():
    """提前统计总案例数（解决进度显示问题）"""
    total = 0
    for subject in SUBJECTS:
        subj_source_dir = os.path.join(SOURCE_DIR, subject)
        if not check_path(subj_source_dir, "dir"):
            continue
        run_dirs = [d for d in os.listdir(subj_source_dir) if
                    check_path(os.path.join(subj_source_dir, d), "dir") and d.startswith("run-")]
        total += len(run_dirs) * len(STATES)
    return total


# ===================== 批量运行主逻辑（整合优化） =====================
if __name__ == "__main__":
    print("=" * 80)
    print("开始批量处理丘脑-皮层偶极子CCG分析（中和版）")
    print(f"被试列表：{SUBJECTS} | 状态：{STATES} | 结果保存目录：{SAVE_DIR}")
    print("=" * 80)

    # 提前统计总案例数（解决进度显示?的问题）
    TOTAL_CASES = get_total_cases()
    processed_cases = 0
    success_cases = 0

    for subject in SUBJECTS:
        print(f"\n{'=' * 60}\n处理被试：{subject}\n{'=' * 60}")

        # 1. 加载混合源空间（每个被试仅加载1次）
        bem_dir = os.path.join(FREESURFER_DIR, subject, "bem")
        mixed_src_fname = os.path.join(bem_dir, f"{subject}-mixed-src.fif")
        if not check_path(mixed_src_fname):
            print(f"❌ {subject} 混合源空间缺失 → 跳过该被试")
            continue
        src = mne.read_source_spaces(mixed_src_fname)
        print(f"✅ 加载{subject}混合源空间完成")

        # 2. 自动读取当前被试的run目录
        subj_source_dir = os.path.join(SOURCE_DIR, subject)
        if not check_path(subj_source_dir, "dir"):
            print(f"❌ {subject} 无STC数据目录 → 跳过")
            continue
        run_dirs = sorted([d for d in os.listdir(subj_source_dir)
                           if check_path(os.path.join(subj_source_dir, d), "dir") and d.startswith("run-")])
        if not run_dirs:
            print(f"❌ {subject} 无有效run目录 → 跳过")
            continue
        print(f"✅ {subject} 检测到有效run：{run_dirs}")

        # 3. 遍历run和状态
        for run in run_dirs:
            for state in STATES:
                processed_cases += 1
                print(f"\n[{processed_cases}/{TOTAL_CASES}] 处理：{subject}-{run}-{state}")
                if process_single_case(subject, run, state, src):
                    success_cases += 1

        # 释放当前被试的src内存
        del src

    # 批量处理总结
    print("\n" + "=" * 80)
    print(f"批量处理完成 | 总案例数：{TOTAL_CASES} | 成功数：{success_cases} | 失败数：{TOTAL_CASES - success_cases}")
    print(f"成功率：{success_cases / TOTAL_CASES * 100:.1f}%" if TOTAL_CASES > 0 else "无案例处理")
    print("=" * 80)