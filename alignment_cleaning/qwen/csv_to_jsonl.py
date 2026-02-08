import pandas as pd
import os

# ================= 設定區 =================
INPUT_CSV = "paul-cleavedata/alignment_cleaning/cometkiwi/alignment_scores_full.csv"   # 你的 CSV 檔案路徑
OUTPUT_JSONL = "paul-cleavedata/alignment_cleaning/qwen/alignment_scores_full.jsonl" # 輸出的 JSONL 檔案路徑
# =========================================

def csv_to_jsonl(input_path, output_path):
    print(f"Reading CSV from: {input_path}")
    
    # 1. 讀取 CSV
    # encoding='utf-8-sig' 是為了處理 Excel 存檔時可能產生的 BOM
    try:
        df = pd.read_csv(input_path, encoding='utf-8-sig')
    except UnicodeDecodeError:
        df = pd.read_csv(input_path, encoding='utf-8')

    print(f"Loaded {len(df)} rows.")
    
    # 檢查欄位是否齊全 (根據你提供的 columns)
    required_cols = ['src', 'mt', 'labse_score', 'type', 'source_file', 'line_idx', 'comet_score']
    for col in required_cols:
        if col not in df.columns:
            print(f"Warning: Column '{col}' not found in CSV. Output might be incomplete.")

    # 2. 轉換並儲存為 JSONL
    # orient='records' 會將每一列轉為一個 dict
    # lines=True 會讓它變成 JSONL (一行一個 json)
    # force_ascii=False 確保中文字不會變成 \uXXXX
    print(f"Converting and saving to: {output_path}")
    df.to_json(output_path, orient='records', lines=True, force_ascii=False)
    
    print("Conversion complete!")

    # 顯示前 3 行看看結果
    print("\n--- Preview (First 3 lines) ---")
    with open(output_path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if i < 3:
                print(line.strip())
            else:
                break

if __name__ == "__main__":
    if os.path.exists(INPUT_CSV):
        csv_to_jsonl(INPUT_CSV, OUTPUT_JSONL)
    else:
        print(f"Error: Input file {INPUT_CSV} not found.")