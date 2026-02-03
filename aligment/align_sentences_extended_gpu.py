def align_sentences_extended_gpu(en_sentences, zh_sentences, threshold=0.60, max_merge_window=4):
    aligned_pairs = []

    # --- 修改點 D: 確保 encode 產出在 GPU 上的 Tensor ---
    # convert_to_tensor=True 會自動根據模型所在的 device 產出 tensor
    en_embeddings = model_labse.encode(en_sentences, convert_to_tensor=True, device=device)
    zh_embeddings = model_labse.encode(zh_sentences, convert_to_tensor=True, device=device)

    i = 0
    j = 0

    while i < len(en_sentences) and j < len(zh_sentences):
        candidates = []

        # --- A. 測試合併 (Loop) ---
        for k in range(1, max_merge_window + 1):

            # 1:k (英文單句 vs 中文合併)
            if j + k <= len(zh_sentences):
                combined_zh_text = "".join(zh_sentences[j : j+k])

                # --- 修改點 E: 讓動態編碼也在 GPU 進行 ---
                emb_comb_zh = model_labse.encode(combined_zh_text, convert_to_tensor=True, device=device, show_progress_bar=False)

                # util.cos_sim 在 GPU tensor 上運算極快
                sim = util.cos_sim(en_embeddings[i], emb_comb_zh).item() # item() 取回數值到 CPU 做邏輯判斷

                candidates.append({
                    "score": sim, "type": f"1:{k}", "i_step": 1, "j_step": k,
                    "en_text": en_sentences[i], "zh_text": combined_zh_text
                })

            # k:1 (英文合併 vs 中文單句)
            if k > 1 and i + k <= len(en_sentences):
                combined_en_text = " ".join(en_sentences[i : i+k])

                # --- 修改點 E (同上) ---
                emb_comb_en = model_labse.encode(combined_en_text, convert_to_tensor=True, device=device, show_progress_bar=False)

                sim = util.cos_sim(emb_comb_en, zh_embeddings[j]).item()

                candidates.append({
                    "score": sim, "type": f"{k}:1", "i_step": k, "j_step": 1,
                    "en_text": combined_en_text, "zh_text": zh_sentences[j]
                })

        # --- B. 測試 Swap (GPU版) ---
        if i + 1 < len(en_sentences) and j + 1 < len(zh_sentences):
            # 這裡的運算全部在 GPU 上發生
            s1 = util.cos_sim(en_embeddings[i], zh_embeddings[j+1]).item()
            s2 = util.cos_sim(en_embeddings[i+1], zh_embeddings[j]).item()

            avg_score = (s1 + s2) / 2

            if min(s1, s2) > threshold - 0.1:
                candidates.append({
                    "score": avg_score, "type": "swap", "i_step": 2, "j_step": 2,
                    "swap_data": [(en_sentences[i], zh_sentences[j+1], s1),
                                  (en_sentences[i+1], zh_sentences[j], s2)]
                })

        # --- C. 決策邏輯 ---
        if not candidates: break
        best_candidate = max(candidates, key=lambda x: x['score'])

        if best_candidate['score'] < threshold:
            # Lookahead logic
            skip_zh_score = 0
            if j + 1 < len(zh_sentences):
                skip_zh_score = util.cos_sim(en_embeddings[i], zh_embeddings[j+1]).item()

            skip_en_score = 0
            if i + 1 < len(en_sentences):
                skip_en_score = util.cos_sim(en_embeddings[i+1], zh_embeddings[j]).item()

            if skip_zh_score > threshold: j += 1
            elif skip_en_score > threshold: i += 1
            else: i += 1; j += 1
            continue

        if best_candidate['type'] == 'swap':
            d1, d2 = best_candidate['swap_data']
            aligned_pairs.append({"en": d1[0], "zh": d1[1], "type": "swap_1", "score": d1[2]})
            aligned_pairs.append({"en": d2[0], "zh": d2[1], "type": "swap_2", "score": d2[2]})
        else:
            aligned_pairs.append({
                "en": best_candidate['en_text'],
                "zh": best_candidate['zh_text'],
                "type": best_candidate['type'],
                "score": best_candidate['score']
            })

        i += best_candidate['i_step']
        j += best_candidate['j_step']

    return aligned_pairs