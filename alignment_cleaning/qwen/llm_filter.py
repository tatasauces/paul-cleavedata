import json
import torch
import re
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

# ================= 設定區 =================
CHECK_THRESHOLD_MIN = 0.55
CHECK_THRESHOLD_MAX = 0.80

MODEL_ID = "Qwen/Qwen3-4B-Instruct-2507"
INPUT_DATA = "paul-cleavedata/alignment_cleaning/qwen/alignment_scores_full.jsonl" 
OUTPUT_FILE = "paul-cleavedata/alignment_cleaning/qwen/final_cleaned_pairs.jsonl"
# =========================================

class TranslationEvaluator:
    def __init__(self, model_name):
        print(f"Loading model: {model_name}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype="auto",
            device_map="auto"
        )
        
        # 定義 System Prompt：賦予角色與明確標準
        self.system_prompt = """You are a professional literary editor performing strict quality control on English-Chinese novel translations.

Your task is to classify whether the Chinese sentence is a valid translation of the English sentence.

**Judgment Criteria:**
1. **Allow Literary/Free Translation (意譯):** In novels, direct translation is often boring. If the Chinese changes the sentence structure or uses idioms but *preserves the core meaning*, it is a "KEEP".
2. **Reject Misalignment/Hallucination:** If the meaning is completely different, numbers don't match, or names are wrong, it is a "DISCARD".

**Output Format:**
You must output a single valid JSON object containing:
- "decision": "KEEP" or "DISCARD"
- "reason": A very short explanation (under 15 words).
"""

        # Few-Shot Examples：這是提升小模型準確率的關鍵
        self.few_shot_examples = """
Here are examples of your judgment logic:

User: 
EN: They’re all in on it.
ZH: 「他們都在。」
Assistant: {"decision": "KEEP", "reason": "Accurate literary translation."}

User:
EN: Because you didn’t know you’d done it.
ZH: 因為你不知道是不是你乾的。
Assistant: {"decision": "KEEP", "reason": "Translate appropriately colloquial (or casual or informal) speech"}

User:
EN: She was driven by curiosity.
ZH: 好奇心驅使著她。
Assistant: {"decision": "KEEP", "reason": "Passive to active voice change is acceptable."}

User:
EN: But the fact of the matter is you are a makeup artist. Technically. Or were—because now you have a ghost makeup artist tapping those keys on your behalf..
ZH: 移民不是笑話，但從技術層面來說，你就是一個擅長編造的藝術家啊。或者說，曾經是，因為你現在已經被一個擅長編造的幽靈藝術家鳩佔鵲巢了。
Assistant: {"decision": "DISCARD", "reason": "Translation quality is great, but not all sentences are translated."}

User:
EN: “It was almost a year ago. You murdered your own wife,” Mayor says, that smug look on his face, that all-knowing, I’m smarter than you look that is making Jerry start to shake with anger.
ZH: 「大概在一年前，你殺了你的妻子。」
Assistant: {"decision": "DISCARD", "reason": "Misaligned with English."}
"""

    def construct_prompt(self, en, zh):
        """建構完整的對話 prompt"""
        user_input = f"Evaluate this pair:\nEN: {en}\nZH: {zh}"
        
        messages = [
            {"role": "system", "content": self.system_prompt},
            # 這裡我們把 few-shot 塞在 user 的前文或 system 中，
            # 對於 Chat 模型，通常把範例放在 system 或第一輪對話中效果最好。
            # 這裡為了簡單，我們把它視為 System instruction 的延伸。
            {"role": "user", "content": self.few_shot_examples + "\n\n" + user_input}
        ]
        return messages

    def parse_output(self, content):
        """解析模型輸出，防止模型輸出 JSON 以外的廢話"""
        try:
            # 嘗試直接解析
            return json.loads(content)
        except json.JSONDecodeError:
            # 如果失敗，使用 Regex 尋找 JSON區塊
            # 尋找 { ... } 結構
            match = re.search(r'\{.*\}', content, re.DOTALL)
            if match:
                try:
                    json_str = match.group(0)
                    # 清理可能存在的 markdown code block 標記
                    json_str = json_str.replace("```json", "").replace("```", "")
                    return json.loads(json_str)
                except:
                    pass
            
            # 如果還是失敗，進行關鍵字暴力判定 (Fallback)
            decision = "DISCARD" # 預設保守策略
            if "KEEP" in content.upper():
                decision = "KEEP"
            return {"decision": decision, "reason": "Parsed via fallback regex"}

    def evaluate(self, en, zh):
        """執行推論"""
        messages = self.construct_prompt(en, zh)
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        
        model_inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)

        # 設定生成參數 (Temperature=0.1 確保穩定性)
        generated_ids = self.model.generate(
            **model_inputs,
            max_new_tokens=128,      # 不需要太長
            temperature=0.1,         # 低溫，減少幻覺
            top_p=0.9,
            do_sample=True
        )
        
        output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist()
        content = self.tokenizer.decode(output_ids, skip_special_tokens=True)
        
        return self.parse_output(content)

# ================= 主程式邏輯 =================

def process_filtering(input_jsonl_path, evaluator):
    """
    讀取 CometKiwi 評分過的檔案，進行 LLM 過濾
    """
    results = []
    
    with open(input_jsonl_path, 'r', encoding='utf-8') as f:
        # 讀取所有行
        lines = f.readlines()
        
    print(f"Processing {len(lines)} sentences...")
    
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f_out:
        for line in tqdm(lines):
            try:
                record = json.loads(line)
                score = record.get("comet_score", 0) # 假設前一步驟有存這個欄位
                en_text = record.get("src", "")
                zh_text = record.get("mt", "")

                # === 策略核心 ===
                
                # 情況 A: 分數很高 -> 直接保留，不浪費 LLM 算力
                if score > CHECK_THRESHOLD_MAX:
                    record['llm_decision'] = "KEEP"
                    record['llm_reason'] = "High confidence score (Auto-Keep)"
                    # 寫入結果
                    json.dump(record, f_out, ensure_ascii=False)
                    f_out.write('\n')
                    continue

                # 情況 B: 分數太低 -> 直接丟棄
                if score < CHECK_THRESHOLD_MIN:
                    # 如果你想留底備查，可以標記為 DISCARD 後寫入
                    # 這裡示範直接過濾掉，不寫入
                    continue 

                # 情況 C: 灰色地帶 (Gray Zone) -> 召喚 LLM 進行審判
                evaluation = evaluator.evaluate(en_text, zh_text)
                
                # 記錄 LLM 的決定
                record['llm_decision'] = evaluation.get("decision", "DISCARD")
                record['llm_reason'] = evaluation.get("reason", "Unknown")
                
                # 只有 LLM 說 KEEP 才寫入
                if record['llm_decision'] == "KEEP":
                    json.dump(record, f_out, ensure_ascii=False)
                    f_out.write('\n')
                else:
                    # 這是被 LLM 殺掉的句子，可以選擇 print 出來 debug
                    # print(f"Discarded: {en_text} -> {zh_text} ({evaluation['reason']})")
                    pass

            except json.JSONDecodeError:
                continue

if __name__ == "__main__":
    # 1. 初始化模型
    evaluator = TranslationEvaluator(MODEL_ID)
    
    # 2. 執行過濾 
    import os
    if os.path.exists(INPUT_DATA):
        process_filtering(INPUT_DATA, evaluator)
    else:
        print(f"Input file {INPUT_DATA} not found.")
        # 測試單句功能
        print("Testing single sentence...")
        res = evaluator.evaluate("He made no answer.", "他沒有回答。")
        print(f"Test Result: {res}")