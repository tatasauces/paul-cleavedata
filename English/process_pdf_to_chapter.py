import pdfplumber
from collections import Counter
import re
import os
import logging

# 抑制 pdfminer.pdffont 的警告訊息
logging.getLogger('pdfminer.pdffont').setLevel(logging.ERROR)

# 設定字體大小閾值
FONT_SIZE_THRESHOLD = 15

def clean_filename(text):
    """
    清理標題以作為合法的檔名 (移除 / \ : * ? " < > |)
    """
    # 移除非法字元
    text = re.sub(r'[\\/*?:"<>|]', "", text)
    # 移除換行符號
    text = text.replace("\n", " ").replace("\r", "")
    return text.strip()

def get_chapter_header(page):
    """
    掃描頁面，尋找連續大於 FONT_SIZE_THRESHOLD 的字元。
    回傳: (標題文字, 標題底端Y座標)
    如果沒找到，回傳 (None, 0)
    """
    large_chars = [c for c in page.chars if c["size"] > FONT_SIZE_THRESHOLD]

    if not large_chars:
        return None, 0

    # 找出這些大字元組成的文字
    header_text = "".join([c["text"] for c in large_chars])

    # 找出標題佔據的區域最底端 (bottom)，內文應從這裡之後開始
    # 我們取所有大字元中最大的 bottom 值
    header_bottom = max([c["bottom"] for c in large_chars])

    return clean_filename(header_text), header_bottom

def save_chapter(output_dir,filename, paragraphs):
    """
    將段落寫入檔案，並過濾浮水印
    """
    if not paragraphs:
        return

    # 確保輸出目錄存在
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    full_path = os.path.join(output_dir, f"{filename}.txt")

    with open(full_path, "w", encoding="utf-8") as f:
        for p in paragraphs:
            # 任務要求：刪去內容字段 "OceanofPDF.com"
            clean_p = p.replace("OceanofPDF.com", "")
            # 如果刪除後只剩空白，則不寫入
            if clean_p.strip():
                f.write(clean_p + "\n")

    print(f"已儲存章節: {full_path} (段落數: {len(paragraphs)})")

def is_new_paragraph_logic(line_obj, prev_line_obj, left_margin_mode, page_width):
    """
    (沿用之前的邏輯) 判斷是否為新段落
    """
    if not prev_line_obj: return True

    current_x0 = line_obj['x0']
    prev_text = prev_line_obj['text'].strip()
    sentence_enders = ('.', '?', '!', '"', '”', '’')
    is_sentence_end = prev_text.endswith(sentence_enders)

    if not is_sentence_end:
        if current_x0 > (left_margin_mode + 50): return True
        return False

    if current_x0 > (left_margin_mode + 10): return True

    estimated_right_margin = page_width - left_margin_mode
    if prev_line_obj['x1'] < (estimated_right_margin - 30): return True

    return False

def process_pdf_to_chapters(pdf_path,output_dir):
    '''
    使用方式
    output_dir = "output_text_EN" 多個.txt
    process_pdf_to_chapters("PAUL CLEAVE/Trust_No_One.pdf",output_dir)
    '''

    current_chapter_name = "Prologue_or_Start" # 預設第一章之前的檔名
    chapter_index = 0 #新增計數器
    current_paragraphs = []

    # 用來跨頁合併段落的 buffer
    buffer_paragraph = ""
    prev_line_obj = None

    # 全域左邊界偵測 (簡化版，只跑前幾頁)
    common_x0 = 0
    with pdfplumber.open(pdf_path) as pdf:
        # --- 預先偵測左邊界 ---
        starts = []
        for p in pdf.pages[4:18]:
            words = p.extract_words()
            if words: starts.append(words[0]['x0'])
        if starts:
            common_x0 = Counter([int(x) for x in starts]).most_common(1)[0][0]

        # --- 開始逐頁處理 ---
        for i, page in enumerate(pdf.pages):
            width = page.width

            # 1.【軌道一】檢查是否有章節標題 (文字大小 > 15)
            header_text, header_bottom = get_chapter_header(page)

            if header_text:
                # 發現新章節！
                # A. 先把"上一章"殘留的 buffer 收尾存入 list
                if buffer_paragraph:
                    current_paragraphs.append(buffer_paragraph.strip())
                    buffer_paragraph = ""
                    prev_line_obj = None

                # B. 寫入上一章的檔案：使用 f-string 將序號格式化為 3 位數 (例如: 000_Prologue, 001_Chapter One)
                numbered_filename = f"{chapter_index:03d}_{current_chapter_name}"
                save_chapter(output_dir,numbered_filename, current_paragraphs)

                chapter_index += 1  # 儲存完畢後，序號加 1，準備給下一個章節標題使用

                # C. 重置狀態，準備開始新章節
                current_chapter_name = header_text
                current_paragraphs = []
                print(f"--- 發現新章節: {header_text} (頁數: {i+1}) ---")

            # 2.【軌道二】提取內文 Words
            # 關鍵：只提取 header_bottom 之後的文字，避免把標題重複抓進內文
            words = page.extract_words(x_tolerance=3, y_tolerance=6)

            # 過濾掉標題區域的字 (只保留 top > header_bottom 的字)
            # 加上一個小緩衝區 (+5) 避免切太齊
            content_words = [w for w in words if w['top'] > header_bottom + 5]

            if not content_words:
                continue

            # 3. 組裝行 (Lines)
            lines = []
            current_line = [content_words[0]]
            for word in content_words[1:]:
                if abs(word['top'] - current_line[-1]['top']) < 10:
                    current_line.append(word)
                else:
                    lines.append(current_line)
                    current_line = [word]
            lines.append(current_line)

            # 4. 段落判斷 (Paragraphs)
            for line in lines:
                line_text = " ".join([w['text'] for w in line])

                # 在這裡也可以先做一次簡單過濾，雖然 save_chapter 也會做
                if "OceanofPDF.com" in line_text:
                    line_text = line_text.replace("OceanofPDF.com", "").strip()
                    if not line_text: continue

                line_x0 = line[0]['x0']
                line_x1 = line[-1]['x1']
                current_line_obj = {'x0': line_x0, 'x1': line_x1, 'text': line_text}

                is_new = is_new_paragraph_logic(current_line_obj, prev_line_obj, common_x0, width)

                if is_new:
                    if buffer_paragraph:
                        current_paragraphs.append(buffer_paragraph.strip())
                    buffer_paragraph = line_text
                else:
                    if buffer_paragraph.endswith("-"):
                        buffer_paragraph = buffer_paragraph[:-1] + line_text
                    else:
                        buffer_paragraph += " " + line_text

                prev_line_obj = current_line_obj

    # 5. 迴圈結束後，別忘了儲存最後一章
    if buffer_paragraph:
        current_paragraphs.append(buffer_paragraph.strip())

    # 儲存最後一章時也加上序號 ---
    numbered_filename = f"{chapter_index:03d}_{current_chapter_name}"
    save_chapter(output_dir,numbered_filename, current_paragraphs)

if __name__=='__main__':
    input_pdf = r"/home/user/paul-cleavedata/BOOK/PAUL CLEAVE_EN/Trust_No_One.pdf"
    OUTPUT_FOLDER = "output_text_EN"

    if not os.path.exists(OUTPUT_FOLDER):
        os.mkdir(OUTPUT_FOLDER)
    
    process_pdf_to_chapters(input_pdf,OUTPUT_FOLDER)
    print("\n所有檔案處理完成！")