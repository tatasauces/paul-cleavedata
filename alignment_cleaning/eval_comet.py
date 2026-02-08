import os
import json
import glob
import torch
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from comet import download_model, load_from_checkpoint
from tqdm import tqdm
from huggingface_hub import login
from dotenv import load_dotenv

# ================= 設定區 =================
INPUT_FOLDER = "/aligment/pairs_sentence"  # 你的 json 檔案存放資料夾
OUTPUT_FILE = "alignment_scores_full.csv" # 儲存所有分數的結果
PLOT_FILE = "score_distribution.png"      # 儲存分佈圖的圖片路徑
BATCH_SIZE = 128  # GPU 顯存越大，可以設越大 (32, 64, 128)
# =========================================

def load_data(folder_path):
    """
    載入資料夾下所有 JSON 檔案，並轉換格式
    """
    json_files = glob.glob(os.path.join(folder_path, "*.json")) # 或 .jsonl
    all_samples = []
    
    print(f"Found {len(json_files)} files in {folder_path}")
    
    for file_path in tqdm(json_files, desc="Loading files"):
        file_name = os.path.basename(file_path)
        with open(file_path, 'r', encoding='utf-8') as f:
            # 假設每個檔案是 JSONL (一行一個 json object)
            # 如果是單個大 JSON list，請改用 json.load(f)
            for line in f:
                if not line.strip(): continue
                try:
                    record = json.loads(line)
                    # 轉換為 COMET 需要的 key: src (來源), mt (機器翻譯/目標)
                    # 同時保留原始 metadata 以便追蹤
                    sample = {
                        "src": record.get("en", ""),
                        "mt": record.get("zh", ""),
                        "labse_score": record.get("score", 0),
                        "type": record.get("type", "unknown"),
                        "source_file": file_name
                    }
                    if sample["src"] and sample["mt"]: # 確保非空
                        all_samples.append(sample)
                except json.JSONDecodeError:
                    continue
                    
    print(f"Total samples loaded: {len(all_samples)}")
    return all_samples

def run_comet_inference(samples):
    """
    執行 CometKiwi 模型推論
    """
    load_dotenv()
    hf_token = os.getenv("HUGGINGFACE_TOKEN")
    if hf_token:
        login(token=hf_token)
    else:
        print("錯誤：找不到 HUGGINGFACE_TOKEN，請檢查 .env 檔案")

    print("Loading CometKiwi model (Quality Estimation)...")
    # 下載並載入無參考模型 (Reference-Free)
    model_path = download_model("Unbabel/wmt22-cometkiwi-da")
    model = load_from_checkpoint(model_path)
    
    # 檢查 GPU
    gpus = 1 if torch.cuda.is_available() else 0
    if gpus > 0:
        print(f"unning on GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("Running on CPU (Might be slow)")

    print(f"Starting inference with Batch Size = {BATCH_SIZE}...")
    
    # model.predict 接受 list of dicts [{'src':..., 'mt':...}]
    # 這是最高效的批次處理方式
    model_output = model.predict(
        samples, 
        batch_size=BATCH_SIZE, 
        gpus=gpus,
        progress_bar=True
    )
    
    return model_output.scores

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
    
    plt.title(f"CometKiwi Score Distribution (N={len(df)})", fontsize=15)
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
    # 1. 載入資料
    data_list = load_data(INPUT_FOLDER)
    if not data_list:
        print("No data found.")
        return

    # 2. 執行推論
    # 為了節省記憶體，我們只傳入需要的欄位給 model
    inference_input = [{"src": d["src"], "mt": d["mt"]} for d in data_list]
    scores = run_comet_inference(inference_input)
    
    # 3. 將分數合併回原始資料
    for i, score in enumerate(scores):
        data_list[i]['comet_score'] = score

    # 4. 轉為 DataFrame 並儲存
    df = pd.DataFrame(data_list)
    
    # 按照分數排序，方便查看低分句
    df = df.sort_values(by="comet_score", ascending=True)
    
    print(f"Saving scores to {OUTPUT_FILE}...")
    df.to_csv(OUTPUT_FILE, index=False, encoding='utf-8-sig') # utf-8-sig 讓 Excel 開啟不亂碼

    # 5. 分析與繪圖
    analyze_and_plot(df, PLOT_FILE)

if __name__ == "__main__":
    main()