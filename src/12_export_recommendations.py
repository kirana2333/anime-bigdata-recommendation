import hashlib
import html
import json
from pathlib import Path

import numpy as np
import pandas as pd
from pyspark.sql import SparkSession
from pyspark.ml.recommendation import ALSModel

TOP_K = 10
USER_LIMIT = 20

with open("output/hybrid_final_meta.json") as f:
    config = json.load(f)

alpha = float(config["best_alpha"])

spark = (
    SparkSession.builder
    .appName("AnimeStrategyExport")
    .master("local[2]")
    .config("spark.driver.memory", "1g")
    .config("spark.ui.enabled", "false")
    .getOrCreate()
)

spark.sparkContext.setLogLevel("ERROR")

# 读取最终训练好的模型
model = ALSModel.load("models/als_hybrid_final")

user_factors = {
    int(r["id"]): np.array(r["features"])
    for r in model.userFactors.collect()
}

item_factors = {
    int(r["id"]): np.array(r["features"])
    for r in model.itemFactors.collect()
}

ratings = pd.read_csv(
    "data/processed/ratings_als_sample.csv"
)

# 保持与最终实验相同的训练集划分
def bucket(user_id, anime_id):
    key = f"hybrid-v1:{int(user_id)}:{int(anime_id)}"
    digest = hashlib.sha256(key.encode()).digest()
    return int.from_bytes(digest[:8], "big") % 100

train_mask = [
    bucket(u, a) < 85
    for u, a in zip(ratings.user_id, ratings.anime_id)
]

train_ratings = ratings.loc[train_mask]

# 选择历史评分较多的20名已学习用户
user_stats = (
    train_ratings.groupby("user_id")
    .size()
    .reset_index(name="count")
    .sort_values(
        ["count", "user_id"],
        ascending=[False, True]
    )
)

users = [
    int(u)
    for u in user_stats.user_id
    if int(u) in user_factors
][:USER_LIMIT]

# 加载最终测试时建立的候选动漫及热门度
popularity = pd.read_csv(
    "output/hybrid_item_popularity.csv"
)

items = sorted(
    int(i)
    for i in popularity.anime_id
    if int(i) in item_factors
)

pop_map = dict(zip(
    popularity.anime_id.astype(int),
    popularity.popularity_score.astype(float)
))

item_array = np.array(items)

item_matrix = np.stack([
    item_factors[i] for i in items
])

pop_scores = np.array([
    pop_map[i] for i in items
])

def normalize(values):
    minimum = values.min()
    maximum = values.max()
    if maximum <= minimum:
        return np.zeros_like(values, dtype=float)
    return (values - minimum) / (maximum - minimum)

records = []

for uid in users:
    # 网页展示时排除数据集中该用户所有已评分作品
    seen = set(
        ratings.loc[
            ratings.user_id == uid, "anime_id"
        ].astype(int)
    )

    mask = ~np.isin(item_array, list(seen))
    candidates = item_array[mask]

    if len(candidates) < TOP_K:
        continue

    als_raw = item_matrix[mask] @ user_factors[uid]
    als_scores = normalize(als_raw)
    popular_scores = pop_scores[mask]

    strategies = {
        "Hybrid": (
            alpha * als_scores
            + (1 - alpha) * popular_scores
        ),
        "ALS": als_scores,
        "Popularity": popular_scores
    }

    for strategy, scores in strategies.items():
        order = np.lexsort(
            (candidates, -scores)
        )[:TOP_K]

        for rank, pos in enumerate(order, 1):
            records.append({
                "user_id": uid,
                "strategy": strategy,
                "rank": rank,
                "anime_id": int(candidates[pos]),
                "score": round(float(scores[pos]), 6)
            })

# 关联动漫名称与题材
anime = pd.read_csv(
    "data/raw/anime.csv",
    usecols=["anime_id", "name", "type", "genre"]
)

anime = anime.drop_duplicates("anime_id")
anime["name"] = (
    anime["name"].fillna("").map(html.unescape)
)

result = pd.DataFrame(records).merge(
    anime, on="anime_id", how="left"
)

result[["name", "type", "genre"]] = result[
    ["name", "type", "genre"]
].fillna("未知")

output_path = Path(
    "output/recommendations_strategy_multiuser.csv"
)

result.to_csv(output_path, index=False)

print("===== 三策略推荐导出完成 =====")
print("用户数量：", result.user_id.nunique())
print("推荐记录数：", len(result))
print("混合权重：", alpha)

print("\n各策略推荐数量：")
print(result.groupby("strategy").size())

print("\n输出文件：", output_path)

spark.stop()
