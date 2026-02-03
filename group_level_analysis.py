#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
组水平统计分析
SOZ + Wodeyar 2024 方法的丘脑-皮层癫痫波传播分析
"""

import os
import numpy as np
import pandas as pd
from scipy import stats

RESULTS_DIR = "/data/shared_home/tlm/Project/MEG-C/results_SOZ_Wodeyar"

def collect_all_results():
    """收集所有被试的结果"""
    results = []

    for subject in os.listdir(RESULTS_DIR):
        subj_dir = os.path.join(RESULTS_DIR, subject)
        if not os.path.isdir(subj_dir):
            continue

        for run in os.listdir(subj_dir):
            run_dir = os.path.join(subj_dir, run)
            if not os.path.isdir(run_dir):
                continue

            for state in os.listdir(run_dir):
                state_dir = os.path.join(run_dir, state)
                npy_file = os.path.join(state_dir, f"{subject}_{run}_{state}_soz_wodeyar.npy")

                if os.path.exists(npy_file):
                    data = np.load(npy_file, allow_pickle=True).item()
                    results.append({
                        'subject': subject,
                        'run': run,
                        'state': state,
                        'di': data.get('di', np.nan),
                        'enhancement': data.get('enhancement', np.nan),
                        'latency': data.get('latency', np.nan),
                        'n_ref': data.get('n_ref', np.nan),
                        'n_target': data.get('n_target', np.nan),
                    })

    return pd.DataFrame(results)

def main():
    print("=" * 80)
    print("组水平统计分析：SOZ + Wodeyar 2024 方法")
    print("=" * 80)

    df = collect_all_results()

    print(f"\n有效分析样本数: {len(df)}")
    print(f"涉及被试数: {df['subject'].nunique()}")
    print(f"被试列表: {', '.join(sorted(df['subject'].unique()))}")

    # 按状态分组
    print("\n" + "=" * 60)
    print("按意识状态分组统计")
    print("=" * 60)

    for state in ['EC', 'EO']:
        state_df = df[df['state'] == state]
        if len(state_df) > 0:
            print(f"\n{state}状态 (n={len(state_df)}):")
            print(f"  DI均值: {state_df['di'].mean():.3f} ± {state_df['di'].std():.3f}")
            print(f"  信号增强: {state_df['enhancement'].mean():.2f}x ± {state_df['enhancement'].std():.2f}x")

            # DI方向统计
            positive = (state_df['di'] > 0).sum()
            negative = (state_df['di'] < 0).sum()
            zero = (state_df['di'] == 0).sum()
            print(f"  SOZ→丘脑: {positive} ({positive/len(state_df)*100:.1f}%)")
            print(f"  丘脑→SOZ: {negative} ({negative/len(state_df)*100:.1f}%)")
            print(f"  双向平衡: {zero} ({zero/len(state_df)*100:.1f}%)")

    # 总体统计
    print("\n" + "=" * 60)
    print("总体统计")
    print("=" * 60)
    print(f"\n总样本数: {len(df)}")
    print(f"DI均值: {df['di'].mean():.3f} ± {df['di'].std():.3f}")
    print(f"信号增强: {df['enhancement'].mean():.2f}x ± {df['enhancement'].std():.2f}x")

    # DI方向统计（总体）
    positive = (df['di'] > 0).sum()
    negative = (df['di'] < 0).sum()
    zero = (df['di'] == 0).sum()
    print(f"\n传播方向分布:")
    print(f"  SOZ→丘脑 (DI>0): {positive} ({positive/len(df)*100:.1f}%)")
    print(f"  丘脑→SOZ (DI<0): {negative} ({negative/len(df)*100:.1f}%)")
    print(f"  双向平衡 (DI=0): {zero} ({zero/len(df)*100:.1f}%)")

    # 按被试统计
    print("\n" + "=" * 60)
    print("按被试统计")
    print("=" * 60)

    for subject in sorted(df['subject'].unique()):
        subj_df = df[df['subject'] == subject]
        print(f"\n{subject} (n={len(subj_df)}):")
        print(f"  DI均值: {subj_df['di'].mean():.3f} ± {subj_df['di'].std():.3f}")
        print(f"  信号增强: {subj_df['enhancement'].mean():.2f}x")

        # 判断主要方向
        mean_di = subj_df['di'].mean()
        if mean_di > 0.1:
            direction = "SOZ→丘脑主导"
        elif mean_di < -0.1:
            direction = "丘脑→SOZ主导"
        else:
            direction = "双向平衡"
        print(f"  主要方向: {direction}")

    # 与零比较的单样本t检验
    print("\n" + "=" * 60)
    print("统计检验")
    print("=" * 60)

    t_stat, p_value = stats.ttest_1samp(df['di'].dropna(), 0)
    print(f"\nDI与零的单样本t检验:")
    print(f"  t = {t_stat:.3f}, p = {p_value:.3f}")

    if p_value < 0.05:
        if df['di'].mean() > 0:
            print(f"  结论: DI显著大于0，整体倾向于SOZ→丘脑传播")
        else:
            print(f"  结论: DI显著小于0，整体倾向于丘脑→SOZ传播")
    else:
        print(f"  结论: DI与0无显著差异，双向平衡")

    # EC vs EO比较
    ec_di = df[df['state'] == 'EC']['di'].dropna()
    eo_di = df[df['state'] == 'EO']['di'].dropna()

    if len(ec_di) > 0 and len(eo_di) > 0:
        t_stat, p_value = stats.ttest_ind(ec_di, eo_di)
        print(f"\nEC vs EO状态DI比较 (独立样本t检验):")
        print(f"  EC DI: {ec_di.mean():.3f} ± {ec_di.std():.3f}")
        print(f"  EO DI: {eo_di.mean():.3f} ± {eo_di.std():.3f}")
        print(f"  t = {t_stat:.3f}, p = {p_value:.3f}")

        if p_value < 0.05:
            print(f"  结论: EC和EO状态的传播方向存在显著差异")
        else:
            print(f"  结论: EC和EO状态的传播方向无显著差异")

    # 保存汇总表
    summary_file = "/data/shared_home/tlm/Project/MEG-C/results_SOZ_Wodeyar/group_summary.csv"
    df.to_csv(summary_file, index=False)
    print(f"\n汇总表已保存至: {summary_file}")

    print("\n" + "=" * 80)
    print("✅ 组水平分析完成！")
    print("=" * 80)

if __name__ == "__main__":
    main()
