import os
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ================= 設定區 =================
INPUT_FILE = "final_cleaned_pairs.jsonl"  # 你的 json 檔案存放資料夾
OUTPUT_FILE = "alignment_scores_full.csv" # 儲存所有分數的結果
PLOT_FILE = "score_distribution.png"      # 儲存分佈圖的圖片路徑
# =========================================

def analyze_and_plot(df, output_img_path):
    """
    進行描述性統計並繪圖
    """
    scores = df['comet_score']
    
    # 1. 描述性統計
    stats = scores.describe(percentiles=[.05, .10, .25, .5, .75, .90, .95])
    print("=== Descriptive Statistics ===")
    print(stats)
    
    # 計算建議閾值 (Mean - 1.5 Std)
    suggested_threshold = stats['mean'] - (1.5 * stats['std'])
    print(f"Suggested Threshold (Mean - 1.5*Std): {suggested_threshold:.4f}")
    
    # 2. 繪圖
    plt.figure(figsize=(12, 6))
    sns.set_style("whitegrid")
    
    # 繪製直方圖與 KDE 曲線
    ax = sns.histplot(scores, bins=100, kde=True, color='skyblue', edgecolor='black', alpha=0.7)
    
    # 標示統計線
    plt.axvline(stats['mean'], color='red', linestyle='--', label=f"Mean: {stats['mean']:.2f}")
    plt.axvline(stats['50%'], color='green', linestyle='-', label=f"Median: {stats['50%']:.2f}")
    plt.axvline(suggested_threshold, color='orange', linestyle=':', linewidth=2, label=f"Sug. Cutoff: {suggested_threshold:.2f}")
    
    plt.title(f"QWEN-CometKiwi Score Distribution (N={len(df)})", fontsize=15)
    plt.xlabel("Quality Score (Direct Assessment)", fontsize=12)
    plt.ylabel("Count", fontsize=12)
    plt.legend()
    
    # 加入統計文字框
    textstr = '\n'.join((
        f"Mean: {stats['mean']:.2f}",
        f"Std:  {stats['std']:.2f}",
        f"Min:  {stats['min']:.2f}",
        f"Max:  {stats['max']:.2f}",
        f"P10:  {stats['10%']:.2f}"
    ))
    props = dict(boxstyle='round', facecolor='white', alpha=0.5)
    ax.text(0.02, 0.95, textstr, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', bbox=props)

    plt.tight_layout()
    plt.savefig(output_img_path, dpi=300)
    print(f"Plot saved to {output_img_path}")

def main():
    # 1. 轉為 DataFrame 並儲存
    df = pd.read_json(INPUT_FILE, lines=True)
    
    # 按照分數排序，方便查看低分句
    df = df.sort_values(by="comet_score", ascending=True)
    
    print(f"Saving scores to {OUTPUT_FILE}...")
    df.to_csv(OUTPUT_FILE, index=False, encoding='utf-8-sig') # utf-8-sig 讓 Excel 開啟不亂碼

    # 2. 分析與繪圖
    analyze_and_plot(df, PLOT_FILE)

if __name__ == "__main__":
    main()