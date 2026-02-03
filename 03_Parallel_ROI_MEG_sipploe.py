#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
并行处理版本的丘脑-皮层CCG分析
充分利用多核CPU资源
"""

import os
import mne
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import partial
import multiprocessing as mp

# ===================== 全局配置 =====================
SUBJECTS = [f"sub-{i:02d}" for i in range(1, 9)]  # sub-01 ~ sub-08
STATES = ["EC", "EO"]

SOURCE_DIR = "/data/shared_home/tlm/Project/MEG-C/source"
FREESURFER_DIR = "/data/shared_home/tlm/data/MEG-C/freesurfer/mri"
SAVE_DIR = "/data/shared_home/tlm/Project/MEG-C/results4_parallel"

# 创建输出目录
os.makedirs(SAVE_DIR, exist_ok=True)

# 丘脑源空间索引映射
THAL_SRC_MAP = {
    "left": {"src_idx": 2, "name": "Left-Thalamus-Proper", "hemi": "lh"},
    "right": {"src_idx": 3, "name": "Right-Thalamus-Proper", "hemi": "rh"}
}

# 导入分析函数
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from SEEG_Cortex_Thalamus_CCG_Analysis import run_spike_propagation_analysis

# ===================== 工具函数 =====================
def check_path(path, path_type="file"):
    """检查文件/目录是否存在"""
    if path_type == "file":
        return os.path.isfile(path)
    elif path_type == "dir":
        return os.path.isdir(path)
    return False


def find_stc_file(run_path, subject, run, state):
    """模糊匹配STC文件"""
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
        return -1, 0, np.zeros(stc_data.shape[1])

    roi_ts = stc_data[roi_verts_local, :]
    dipole_activity = np.max(np.abs(roi_ts), axis=1)
    max_idx_in_roi = np.argmax(dipole_activity)
    dipole_idx_local = roi_verts_local[max_idx_in_roi]
    dipole_max_amp = dipole_activity[max_idx_in_roi]
    dipole_ts = stc_data[dipole_idx_local, :]
    return dipole_idx_local, dipole_max_amp, dipole_ts


# ===================== 单个案例处理函数 =====================
def process_single_case_wrapper(subject, run, state, src, vertno_map, 
                                 source_dir, freesurfer_dir, save_dir):
    """
    处理单个被试-单个run-单个状态的包装函数
    用于并行处理
    """
    try:
        # 1. 定位STC文件
        run_path = os.path.join(source_dir, subject, run)
        stc_fname = find_stc_file(run_path, subject, run, state)
        if not stc_fname or not check_path(stc_fname):
            return {'status': 'error', 'msg': f'{subject}-{run}-{state}: STC文件缺失', 
                    'subject': subject, 'run': run, 'state': state}

        # 2. 读取STC数据
        stc = mne.read_source_estimate(stc_fname)
        times = stc.times
        sfreq = 1 / (times[1] - times[0])
        
        # 3. 提取皮层ROI（aparc分区）
        labels_cortex = mne.read_labels_from_annot(subject, parc="aparc", 
                                                   subjects_dir=freesurfer_dir)
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

        # 5. 构建同侧丘脑ROI
        thal_config = THAL_SRC_MAP[cortex_side]
        thal_verts_global = src[thal_config["src_idx"]]["vertno"]
        thal_verts_local = [vertno_map[v] for v in thal_verts_global if v in vertno_map]
        
        if len(thal_verts_local) == 0:
            return {'status': 'error', 'msg': f'{subject}-{run}-{state}: 丘脑无有效偶极子',
                    'subject': subject, 'run': run, 'state': state}

        thal_label = mne.Label(vertices=thal_verts_global, hemi=thal_config["hemi"], 
                               name=thal_config["name"], subject=subject)
        label_ts_thal = mne.extract_label_time_course([stc], [thal_label], src, 
                                                       mode="mean", allow_empty=True)[0][0]

        # 6. 找最活跃偶极子
        cortex_dipole_idx, cortex_dipole_amp, cortex_dipole_ts = find_most_active_dipole(
            most_active_cortex["verts_local"], stc.data
        )
        thal_dipole_idx, thal_dipole_amp, thal_dipole_ts = find_most_active_dipole(
            thal_verts_local, stc.data
        )

        # 7. 运行CCG分析并保存结果
        tag = f"{subject}_{run}_{state}"
        case_save_dir = os.path.join(save_dir, subject, run, state)
        os.makedirs(case_save_dir, exist_ok=True)
        
        ccg_results = run_spike_propagation_analysis(cortex_dipole_ts, thal_dipole_ts, 
                                                     sfreq, tag=tag, save_dir=case_save_dir)
        
        if ccg_results is None:
            return {'status': 'warning', 'msg': f'{subject}-{run}-{state}: 棘波数量过少',
                    'subject': subject, 'run': run, 'state': state}

        # 保存结果
        np.save(os.path.join(case_save_dir, f"{tag}_ccg_results.npy"), ccg_results)
        
        return {
            'status': 'success',
            'subject': subject,
            'run': run,
            'state': state,
            'cortex_roi': most_active_cortex['name'],
            'cortex_amp': float(cortex_dipole_amp),
            'thal_amp': float(thal_dipole_amp),
            'di': float(ccg_results.get('di', 0)),
            'latency': ccg_results.get('latency'),
            'enhancement': float(ccg_results.get('enhancement', 0)),
            'n_spikes': int(ccg_results.get('n_spikes', 0))
        }

    except Exception as e:
        return {'status': 'error', 'msg': f'{subject}-{run}-{state}: {str(e)[:100]}',
                'subject': subject, 'run': run, 'state': state}


# ===================== 主程序 =====================
def main():
    print("=" * 80)
    print("并行处理丘脑-皮层偶极子CCG分析")
    print(f"被试列表: {SUBJECTS} | 状态: {STATES}")
    print(f"结果保存目录: {SAVE_DIR}")
    print(f"可用CPU核心数: {mp.cpu_count()}")
    print("=" * 80)

    # 统计总任务数
    all_tasks = []
    for subject in SUBJECTS:
        subj_source_dir = os.path.join(SOURCE_DIR, subject)
        if not check_path(subj_source_dir, "dir"):
            continue
        run_dirs = [d for d in os.listdir(subj_source_dir)
                    if check_path(os.path.join(subj_source_dir, d), "dir") 
                    and d.startswith("run-")]
        for run in run_dirs:
            for state in STATES:
                all_tasks.append((subject, run, state))

    total_tasks = len(all_tasks)
    print(f"\n总任务数: {total_tasks}")
    print(f"建议并行进程数: {min(mp.cpu_count(), total_tasks)}")

    # 对每个被试加载源空间（复用）
    for subject in SUBJECTS:
        print(f"\n{'=' * 60}\n处理被试: {subject}\n{'=' * 60}")

        # 1. 加载混合源空间
        bem_dir = os.path.join(FREESURFER_DIR, subject, "bem")
        mixed_src_fname = os.path.join(bem_dir, f"{subject}-mixed-src.fif")
        if not check_path(mixed_src_fname):
            print(f"❌ {subject} 混合源空间缺失 → 跳过该被试")
            continue
        src = mne.read_source_spaces(mixed_src_fname)
        print(f"✅ 加载{subject}混合源空间完成")

        # 2. 构建顶点映射
        vertno_map = build_vertno_map(src)

        # 3. 获取当前被试的run列表
        subj_source_dir = os.path.join(SOURCE_DIR, subject)
        if not check_path(subj_source_dir, "dir"):
            print(f"❌ {subject} 无STC数据目录 → 跳过")
            continue
        run_dirs = sorted([d for d in os.listdir(subj_source_dir)
                           if check_path(os.path.join(subj_source_dir, d), "dir") 
                           and d.startswith("run-")])
        if not run_dirs:
            print(f"❌ {subject} 无有效run目录 → 跳过")
            continue

        # 4. 准备该被试的所有任务
        subject_tasks = []
        for run in run_dirs:
            for state in STATES:
                subject_tasks.append((run, state))

        print(f"✅ {subject} 检测到有效runs: {run_dirs}")
        print(f"   任务数: {len(subject_tasks)} (runs × states)")

        # 5. 并行处理该被试的所有任务
        # 使用部分函数固定参数
        process_func = partial(
            process_single_case_wrapper,
            subject=subject,
            src=src,
            vertno_map=vertno_map,
            source_dir=SOURCE_DIR,
            freesurfer_dir=FREESURFER_DIR,
            save_dir=SAVE_DIR
        )

        # 确定并行进程数（不超过任务数和CPU核心数）
        n_workers = min(len(subject_tasks), mp.cpu_count(), 32)  # 最多32个并行
        print(f"   并行进程数: {n_workers}")

        success_count = 0
        warning_count = 0
        error_count = 0

        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            # 提交所有任务
            future_to_task = {
                executor.submit(process_func, run, state): (run, state)
                for run, state in subject_tasks
            }

            # 收集结果
            for future in as_completed(future_to_task):
                run, state = future_to_task[future]
                try:
                    result = future.result()
                    if result['status'] == 'success':
                        success_count += 1
                        print(f"  ✅ {run}-{state}: DI={result['di']:.3f}, "
                              f"Enhancement={result['enhancement']:.2f}x")
                    elif result['status'] == 'warning':
                        warning_count += 1
                        print(f"  ⚠️ {run}-{state}: {result['msg']}")
                    else:
                        error_count += 1
                        print(f"  ❌ {run}-{state}: {result['msg']}")
                except Exception as e:
                    error_count += 1
                    print(f"  ❌ {run}-{state}: 处理异常 - {str(e)[:50]}")

        print(f"\n{subject} 完成: 成功={success_count}, 警告={warning_count}, 错误={error_count}")
        
        # 释放内存
        del src

    # 最终总结
    print("\n" + "=" * 80)
    print("批量处理完成！")
    print(f"结果保存目录: {SAVE_DIR}")
    print("=" * 80)


if __name__ == "__main__":
    main()
