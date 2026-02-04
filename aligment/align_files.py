import os

def create_file_pairs(dir_en, dir_zh):
    """
    在兩個資料夾中尋找序號從 001 到 038 的對應檔案，
    並將它們的完整路徑配對。

    Args:
        dir_en (str): 英文檔案的資料夾路徑。
        dir_zh (str): 中文檔案的資料夾路徑。

    Returns:
        list: 一個包含 (英文檔案路徑, 中文檔案路徑) 元組的列表。
    """
    files_en = os.listdir(dir_en)
    files_zh = os.listdir(dir_zh)
    
    # 排序以確保順序正確
    files_en.sort()
    files_zh.sort()

    paired_files = []
    # 使用 range(1, 39) 來產生 001 到 038 的序號
    for i in range(1, 39):
        prefix = f"{i:03d}_"
        
        # 尋找對應的檔案
        # 使用 next() 搭配生成器表達式來尋找檔案，如果找不到則返回 None
        file_en = next((f for f in files_en if f.startswith(prefix)), None)
        file_zh = next((f for f in files_zh if f.startswith(prefix)), None)
        
        if file_en and file_zh:
            # 建立完整的檔案路徑並加入 list
            path_en = os.path.join(dir_en, file_en)
            path_zh = os.path.join(dir_zh, file_zh)
            paired_files.append((path_en, path_zh))
        else:
            print(f"警告: 找不到以 '{prefix}' 為開頭的對應檔案。")
            
    return paired_files



def split_sentences_spacy(nlp_en,nlp_zh,text, lang='en'):
    """
    使用 Spacy 進行斷句
    args:
    nlp_en : spacy.load("en_core_web_sm")
    nlp_zh : spacy.load("zh_core_web_sm")

    return:
    list of sentences
    """
    if not text or not text.strip():
        return []

    if lang == 'en':
        doc = nlp_en(text)
    else:
        doc = nlp_zh(text)

    # 過濾掉過短的句子或純符號
    return [sent.text.strip() for sent in doc.sents if len(sent.text.strip()) > 1]

def process_chapter_alignment(nlp_en,nlp_zh,en_chapter_path, zh_chapter_path, output_path,align_sentences_function,model,device):
    """
    執行分層對齊：章節 -> 段落 -> 句子

    align_sentences_function 可以設定成align_sentences_extended_gpu()或align_sentences_extended()
    """
    # 1. 讀取檔案 (假設一行一段落)
    with open(en_chapter_path, 'r', encoding='utf-8') as f:
        en_paragraphs = [line.strip() for line in f if line.strip()]

    with open(zh_chapter_path, 'r', encoding='utf-8') as f:
        zh_paragraphs = [line.strip() for line in f if line.strip()]

    print(f"Loaded: {len(en_paragraphs)} EN paragraphs, {len(zh_paragraphs)} ZH paragraphs.")

    # ---------------------------------------------------------
    # 第一階段：段落級對齊 (Paragraph Alignment)
    # ---------------------------------------------------------
    print("Stage 1: Aligning Paragraphs...")
    # 直接複用對齊函數，輸入是段落列表
    # 段落合併通常不會超過 3 段，所以 window 設小一點節省時間
    aligned_paragraphs = align_sentences_function(
        model,
        device,
        en_paragraphs,
        zh_paragraphs,
        threshold=0.50, # 段落相似度通常比句子低一點，因為雜訊多，設低一點
        max_merge_window=3
    )

    print(f"Paragraph alignment done. Found {len(aligned_paragraphs)} pairs.")

    # ---------------------------------------------------------
    # 第二階段：句子級對齊 (Sentence Alignment)
    # ---------------------------------------------------------
    print("Stage 2: Aligning Sentences within Paragraphs...")

    final_sentence_pairs = []

    for para_pair in aligned_paragraphs:
        # 取得配對好的段落文本
        p_en_text = para_pair['en']
        p_zh_text = para_pair['zh']

        # 使用 Spacy 斷句
        sents_en = split_sentences_spacy(nlp_en,nlp_zh,p_en_text, 'en')
        sents_zh = split_sentences_spacy(nlp_en,nlp_zh,p_zh_text, 'zh')

        # 如果任一方斷句後為空，跳過
        if not sents_en or not sents_zh:
            continue

        # 在這個小範圍內進行句對齊
        # 這裡需要高精度，threshold 設高，並開啟 1:4 合併
        sents_pairs = align_sentences_function(
            model,
            device,
            sents_en,
            sents_zh,
            threshold=0.65,
            max_merge_window=4
        )

        # 收集結果，並加上來源段落的 metadata (這對 debug 很有用)
        for sp in sents_pairs:
            sp['source_para_score'] = para_pair['score'] # 記錄這句來自哪個可信度的段落
            final_sentence_pairs.append(sp)

    # ---------------------------------------------------------
    # 輸出結果
    # ---------------------------------------------------------
    print(f"Total sentence pairs aligned: {len(final_sentence_pairs)}")

    # 寫入 JSONL 或 TXT
    import json
    with open(output_path, 'w', encoding='utf-8') as f:
        for pair in final_sentence_pairs:
            json.dump(pair, f, ensure_ascii=False)
            f.write('\n')