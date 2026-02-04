import os
import re

# --------------------------------------------------------
import torch
import spacy
# 設定 Device 
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Running on: {device}")

# 啟用 SpaCy GPU (必須在 load 模型之前執行) ---
if device == "cuda":
    spacy.prefer_gpu()

# 載入 SpaCy 模型 split_sentences_spacy()傳入參數模型
print("Loading SpaCy...")
nlp_en = spacy.load("en_core_web_sm") # 或 trf 版本
nlp_zh = spacy.load("zh_core_web_sm") # 或 trf 版本

# --------------------------------------------------------
from sentence_transformers import SentenceTransformer, util

# 將 LaBSE 模型搬移到 GPU ---
print("Loading LaBSE...")
model = SentenceTransformer('sentence-transformers/LaBSE')
model.to(device) # 關鍵：移動模型權重到 GPU

# --------------------------------------------------------
from align_files import create_file_pairs,split_sentences_spacy,process_chapter_alignment

# 對齊段落、語句是否使用GPU
if device == "cuda":
    #使用GPU
    from align_sentences_extended_gpu import align_sentences_extended_gpu
    align_sentences = align_sentences_extended_gpu
else:
    # 使用CPU
    from align_sentences_extended import align_sentences_extended
    align_sentences = align_sentences_extended
    



if __name__ == "__main__":
    """
    使用方式為放入對應的中英文章節後呼叫 process_chapter_alignment()
    1. 建立存放json的資料夾
    2. 迭代運行
    """

    dir_path = "pairs_sentence"
    if not os.path.exists(dir_path):
        os.mkdir(dir_path)

    EN_dir = r'/paul-cleavedata/English/output_text_EN'
    ZH_dir = r'/paul-cleavedata/Chinese/output_text_ZH'
    
    '''
    兩份資料夾中要處裡的檔案開頭序號為001_,...,038_
    找出對應檔案，然後zip餵入process_chapter_alignment()
    '''
    book_chapter_pairs = create_file_pairs(EN_dir, ZH_dir)

    for i, (en_chapter_path, zh_chapter_path) in enumerate(book_chapter_pairs):
        output_file_name = os.path.join(dir_path, f'aligned_ch{i}.jsonl')
        process_chapter_alignment(
            nlp_en=nlp_en,
            nlp_zh=nlp_zh,
            en_chapter_path=en_chapter_path, 
            zh_chapter_path=zh_chapter_path, 
            output_path=output_file_name, 
            align_sentences_function=align_sentences,
            model= model,
            device=device
        )