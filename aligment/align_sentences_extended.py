def align_sentences_extended(en_sentences, zh_sentences, threshold=0.60, max_merge_window=4):
    """
    支援 1:N 和 N:1 (N最大為 max_merge_window) 的合併測試，以及 2:2 交叉亂序 (Swap)。
    預設 max_merge_window=4，即支援 1:4 和 4:1。
    """
    aligned_pairs = []

    # 預先計算 Embedding (轉換為 Tensor 以利用 GPU 加速計算)
    # 注意：這裡只計算單句的 embedding，合併句會在迴圈中動態計算
    en_embeddings = model.encode(en_sentences, convert_to_tensor=True)
    zh_embeddings = model.encode(zh_sentences, convert_to_tensor=True)

    i = 0
    j = 0

    while i < len(en_sentences) and j < len(zh_sentences):
        candidates = [] # 用來存儲所有可能的對齊策略 [score, type, en_step, zh_step, data_dict]

        # --- A. 測試所有合併情況 (1:N 和 N:1) ---
        # 範圍從 1 到 max_merge_window (預設 1~4)
        for k in range(1, max_merge_window + 1):

            # 情況 1: 1句英文 對 k句中文 (1:k)
            if j + k <= len(zh_sentences):
                # 這裡需要動態合併文本並編碼
                # 簡單用空格連接，實際可根據標點優化
                combined_zh_text = "".join(zh_sentences[j : j+k])
                emb_comb_zh = model.encode(combined_zh_text, convert_to_tensor=True)

                sim = util.cos_sim(en_embeddings[i], emb_comb_zh).item()
                candidates.append({
                    "score": sim,
                    "type": f"1:{k}",
                    "i_step": 1,
                    "j_step": k,
                    "en_text": en_sentences[i],
                    "zh_text": combined_zh_text
                })

            # 情況 2: k句英文 對 1句中文 (k:1)
            # 注意：當 k=1 時，這與上面的 1:1 重複，為了邏輯簡單我們允許重複計算，取 max 沒影響
            if k > 1 and i + k <= len(en_sentences):
                combined_en_text = " ".join(en_sentences[i : i+k])
                emb_comb_en = model.encode(combined_en_text, convert_to_tensor=True)

                sim = util.cos_sim(emb_comb_en, zh_embeddings[j]).item()
                candidates.append({
                    "score": sim,
                    "type": f"{k}:1",
                    "i_step": k,
                    "j_step": 1,
                    "en_text": combined_en_text,
                    "zh_text": zh_sentences[j]
                })

        # --- B. 測試交叉亂序 (Swap Check) ---
        # 僅檢查 2x2 的互換 (E1->C2, E2->C1)
        if i + 1 < len(en_sentences) and j + 1 < len(zh_sentences):
            # E_i vs C_{j+1}
            s1 = util.cos_sim(en_embeddings[i], zh_embeddings[j+1]).item()
            # E_{i+1} vs C_j
            s2 = util.cos_sim(en_embeddings[i+1], zh_embeddings[j]).item()

            avg_score = (s1 + s2) / 2

            # 只有當兩者都達到一定水準，才視為 Swap (避免一個極高一個極低拉高平均)
            # 這裡加一個 min check 讓 swap 條件嚴格一點
            if min(s1, s2) > threshold - 0.1:
                candidates.append({
                    "score": avg_score,
                    "type": "swap",
                    "i_step": 2,
                    "j_step": 2,
                    # Swap 比較特殊，我們會輸出兩筆
                    "swap_data": [
                        (en_sentences[i], zh_sentences[j+1], s1),
                        (en_sentences[i+1], zh_sentences[j], s2)
                    ]
                })

        # --- C. 決策邏輯 ---
        if not candidates:
            break # 邊界保護

        # 找出分數最高的候選者
        best_candidate = max(candidates, key=lambda x: x['score'])

        # 檢查是否過閾值
        if best_candidate['score'] < threshold:
            # 策略：如果都不匹配，判定為某一方有多餘句子
            # 這裡採用的策略是：嘗試跳過中文 (中文常有額外語氣句)
            # 進階策略可以做 Lookahead (看 i+1, j 跟 i, j+1 誰比較合)

            # 這裡示範簡單的 Lookahead Check
            skip_zh_score = 0
            if j + 1 < len(zh_sentences):
                skip_zh_score = util.cos_sim(en_embeddings[i], zh_embeddings[j+1]).item()

            skip_en_score = 0
            if i + 1 < len(en_sentences):
                skip_en_score = util.cos_sim(en_embeddings[i+1], zh_embeddings[j]).item()

            if skip_zh_score > threshold:
                j += 1 # 認定中文多了一句，跳過中文
            elif skip_en_score > threshold:
                i += 1 # 認定英文多了一句，跳過英文
            else:
                # 雙方都無法匹配，同時跳過 (避免死循環)
                i += 1
                j += 1
            continue

        # 執行最佳匹配
        if best_candidate['type'] == 'swap':
            # 處理 Swap
            d1, d2 = best_candidate['swap_data']
            aligned_pairs.append({"en": d1[0], "zh": d1[1], "type": "swap_1", "score": d1[2]})
            aligned_pairs.append({"en": d2[0], "zh": d2[1], "type": "swap_2", "score": d2[2]})
        else:
            # 處理 Merge (1:1, 1:2, ..., 4:1)
            aligned_pairs.append({
                "en": best_candidate['en_text'],
                "zh": best_candidate['zh_text'],
                "type": best_candidate['type'],
                "score": best_candidate['score']
            })

        # 移動指針
        i += best_candidate['i_step']
        j += best_candidate['j_step']

    return aligned_pairs