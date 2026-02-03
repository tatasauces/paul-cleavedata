import pdfplumber
import os
import re

# ==========================================
# 1. 配置區域 (Configuration)
# ==========================================

# 段落間距閾值 (已確認)
GAP_THRESHOLD = 10

# 特殊檔案內容處理 (不提取，直接寫入)
# Key: 檔名開頭 (前三碼), Value: (完整檔名識別用, 硬編碼內容)
SPECIAL_CONTENT_RULES = {
    "022": ("022_搞砸婚禮的當天", "今天是星期天。現在是凌晨一點。這意味著搞砸婚禮的那天已經過去了,新的一天開始了。")
}

# 去除開頭段落數規則
# Key: 檔名開頭 (前三碼), Value: 去除的段落數量
SKIP_PARAGRAPH_RULES = {
    # 去除 6 段
    "001": 6,

    # 去除 3 段
    "015": 3, "017": 3, "024": 3, "030": 3, "032": 3, "035": 3, "038": 3,

    # 去除 5 段 (除了上述規則外的預設值也是 5，但 036 明確指定了)
    "036": 5
}

# 預設去除段落數 (其餘檔案)
DEFAULT_SKIP_COUNT = 5

# ==========================================
# 2. 核心功能函數
# ==========================================

def extract_chinese_by_spacing_filtered(pdf_path, paragraph_gap_threshold=10):
    """
    提取中文段落，並去除頁碼 (n/m 格式)
    """
    all_paragraphs = []
    buffer_paragraph = ""
    prev_line_bottom = None

    # 頁碼的正則表達式: 數字 + 斜線 + 數字 (允許中間有空格)
    # 例如: "1/200", "1 / 200", "15/30"
    page_num_pattern = re.compile(r'^\d+\s*/\s*\d+$')

    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            words = page.extract_words(x_tolerance=5, y_tolerance=3, keep_blank_chars=False)

            if not words: continue

            # --- 過濾頁碼邏輯 ---
            # 策略：檢查頁面最底部(最後幾個物件)是否符合頁碼格式
            # 為了保險，我們過濾掉所有單獨成行且符合 n/m 格式的文字
            filtered_words = []
            for w in words:
                text = w['text'].strip()
                # 如果該字串完全符合頁碼格式，則跳過 (視為頁碼)
                if page_num_pattern.match(text):
                    continue
                filtered_words.append(w)
            words = filtered_words

            if not words: continue
            # ------------------

            # 組裝行 (Lines)
            lines = []
            current_line_words = [words[0]]
            for word in words[1:]:
                if abs(word['top'] - current_line_words[-1]['top']) < 5:
                    current_line_words.append(word)
                else:
                    lines.append(current_line_words)
                    current_line_words = [word]
            lines.append(current_line_words)

            # 逐行分析間距 (Gap)
            for line_words in lines:
                line_text = "".join([w['text'] for w in line_words])
                line_top = min([w['top'] for w in line_words])
                line_bottom = max([w['bottom'] for w in line_words])

                is_new_paragraph = False

                if prev_line_bottom is not None:
                    gap = line_top - prev_line_bottom
                    if gap > paragraph_gap_threshold:
                        is_new_paragraph = True

                if is_new_paragraph:
                    if buffer_paragraph:
                        all_paragraphs.append(buffer_paragraph)
                    buffer_paragraph = line_text
                else:
                    buffer_paragraph += line_text

                prev_line_bottom = line_bottom

            prev_line_bottom = None # 換頁重置

    if buffer_paragraph:
        all_paragraphs.append(buffer_paragraph)

    return all_paragraphs

def get_skip_count(filename):
    """根據檔名決定要跳過前幾個段落"""
    prefix = filename[:3] # 取得前三碼 (例如 "015")

    if prefix in SKIP_PARAGRAPH_RULES:
        return SKIP_PARAGRAPH_RULES[prefix]

    return DEFAULT_SKIP_COUNT

def process_all_files():
    # 建立輸出目錄
    if not os.path.exists(OUTPUT_FOLDER):
        os.makedirs(OUTPUT_FOLDER)

    # 取得 PDF 檔案列表並排序
    pdf_files = [f for f in os.listdir(INPUT_FOLDER) if f.lower().endswith('.pdf')]
    pdf_files.sort() # 確保依照數字順序處理

    print(f"找到 {len(pdf_files)} 個 PDF 檔案，開始處理...\n")

    for filename in pdf_files:
        prefix = filename[:3]
        file_base_name = os.path.splitext(filename)[0] # 去除 .pdf
        output_path = os.path.join(OUTPUT_FOLDER, f"{file_base_name}.txt")
        pdf_path = os.path.join(INPUT_FOLDER, filename)

        paragraphs_to_write = []

        # --- 規則 A: 特殊檔案直接寫入 ---
        if prefix in SPECIAL_CONTENT_RULES:
            # 確認檔名是否包含特定關鍵字 (多重確認)
            target_name_part = SPECIAL_CONTENT_RULES[prefix][0]
            if target_name_part in filename:
                content = SPECIAL_CONTENT_RULES[prefix][1]
                paragraphs_to_write = [content]
                print(f"[{filename}] -> 特殊檔案，寫入指定內容")
            else:
                # 如果編號是 022 但檔名不對，則走一般流程(或報錯)，這裡假設走一般流程
                print(f"[{filename}] -> 編號特殊但檔名不匹配，走一般流程")
                raw_paragraphs = extract_chinese_by_spacing_filtered(pdf_path, GAP_THRESHOLD)
                skip_n = get_skip_count(filename)
                paragraphs_to_write = raw_paragraphs[skip_n:]

        # --- 規則 B: 一般檔案提取並刪減 ---
        else:
            # 1. 提取段落 (含去除頁碼功能)
            raw_paragraphs = extract_chinese_by_spacing_filtered(pdf_path, GAP_THRESHOLD)

            # 2. 決定去除行數
            skip_n = get_skip_count(filename)

            # 3. 執行去除 (Slicing)
            if len(raw_paragraphs) > skip_n:
                paragraphs_to_write = raw_paragraphs[skip_n:]
            else:
                # 如果段落數少於要去除的數量，則寫入空檔或保留最後一段(視需求)
                # 這裡設定為寫入空內容
                paragraphs_to_write = []
                print(f"Warning: [{filename}] 段落數 ({len(raw_paragraphs)}) 少於需去除數 ({skip_n})")

            print(f"[{filename}] -> 去除前 {skip_n} 段 (原始 {len(raw_paragraphs)} -> 剩餘 {len(paragraphs_to_write)})")

        # --- 寫入檔案 (一行一段) ---
        with open(output_path, "w", encoding="utf-8") as f:
            for p in paragraphs_to_write:
                f.write(p + "\n")

# ==========================================
# 3. 執行
# ==========================================

if __name__ == "__main__":
    # 請確保當前目錄下有 'input_pdfs' 資料夾並放入 PDF

    # 輸入與輸出資料夾
    INPUT_FOLDER = r"/home/user/paul-cleavedata/BOOK/PAUL CLEAVE_ZH"
    OUTPUT_FOLDER = "output_text_ZH" # 輸出的 TXT 資料夾名稱

    if os.path.exists(INPUT_FOLDER):
        process_all_files()
        print("\n所有檔案處理完成！")
    else:
        print(f"錯誤: 找不到輸入資料夾 '{INPUT_FOLDER}'")