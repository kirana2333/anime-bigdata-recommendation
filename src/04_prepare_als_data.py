import pandas as pd
from pathlib import Path

input_file = "data/raw/rating.csv"
output_file = "data/processed/ratings_als_sample.csv"

samples = []
total = 0
valid_total = 0

# 分块读取，避免一次加载 780 万条数据
for chunk in pd.read_csv(input_file, chunksize=200000):
    total += len(chunk)

    # 仅保留 1～10 分的有效评分
    valid = chunk[chunk["rating"].between(1, 10)]
    valid_total += len(valid)

    # 首次训练：使用前 1000 个用户的数据
    sample = valid[valid["user_id"] <= 1000]

    if not sample.empty:
        samples.append(sample)

ratings = pd.concat(samples, ignore_index=True)

# 同一用户对同一动漫的重复评分合并
ratings = (
    ratings
    .groupby(["user_id", "anime_id"], as_index=False)
    ["rating"].mean()
)

# 保留至少有 10 条有效评分的用户
user_counts = ratings.groupby("user_id").size()
active_users = user_counts[user_counts >= 10].index

ratings = ratings[
    ratings["user_id"].isin(active_users)
]

if ratings.empty:
    raise RuntimeError("筛选结果为空，需要调整采样条件")

Path(output_file).parent.mkdir(parents=True, exist_ok=True)
ratings.to_csv(output_file, index=False)

print("===== ALS 数据准备完成 =====")
print("原始评分记录：", total)
print("有效评分记录：", valid_total)
print("训练样本记录：", len(ratings))
print("样本用户数量：", ratings["user_id"].nunique())
print("样本动漫数量：", ratings["anime_id"].nunique())
print("输出文件：", output_file)
