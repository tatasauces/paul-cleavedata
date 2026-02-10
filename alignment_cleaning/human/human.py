# 隨機抽樣 comet_score小於0.75 以下的數據共50筆
import pandas as pd

path = "/home/user/paul-cleavedata/alignment_cleaning/qwen/alignment_scores_full.csv"

df = pd.read_csv(path)
df = df.loc[df['comet_score'] <= 0.75,["src","mt","comet_score","source_file","line_idx"]]
df = df.sample(n=100,random_state=42)
df.to_csv("random_sample.csv", index=False, encoding='utf-8-sig')