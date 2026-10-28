import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from pyspark.sql import SparkSession
from pyspark.ml.recommendation import ALS

TOP_K = 10
MIN_RATINGS = 20
POSITIVE_THRESHOLD = 8
USER_LIMIT = 100

# alpha=0 表示纯热门推荐
# alpha=1 表示纯 ALS 推荐
WEIGHTS = [0.0, 0.25, 0.5, 0.75, 1.0]

Path("output").mkdir(exist_ok=True)

# 1. 读取评分数据
ratings = pd.read_csv(
    "data/processed/ratings_als_sample.csv"
)

# 2. 稳定划分训练、验证、测试数据
def get_bucket(user_id, anime_id):
    key = f"hybrid-v1:{int(user_id)}:{int(anime_id)}"
    digest = hashlib.sha256(key.encode()).digest()
    return int.from_bytes(digest[:8], "big") % 100

buckets = np.array([
    get_bucket(u, a)
    for u, a in zip(ratings.user_id, ratings.anime_id)
])

train_pd = ratings.loc[buckets < 70].copy()
valid_pd = ratings.loc[
    (buckets >= 70) & (buckets < 85)
].copy()

print("===== 数据划分 =====")
print("训练集：", len(train_pd))
print("验证集：", len(valid_pd))
print("保留测试集：", int((buckets >= 85).sum()))

# 3. 启动 Spark
spark = (
    SparkSession.builder
    .appName("AnimeHybridValidation")
    .master("local[2]")
    .config("spark.driver.memory", "1g")
    .config("spark.sql.shuffle.partitions", "4")
    .config("spark.default.parallelism", "2")
    .config("spark.ui.enabled", "false")
    .getOrCreate()
)
spark.sparkContext.setLogLevel("ERROR")

train_rows = [
    (int(u), int(a), float(r))
    for u, a, r in train_pd[
        ["user_id", "anime_id", "rating"]
    ].itertuples(index=False, name=None)
]

train = spark.createDataFrame(
    train_rows,
    "user_id INT, anime_id INT, rating FLOAT"
)

# 4. 训练 ALS
print("\n===== 训练 ALS =====")

als = ALS(
    userCol="user_id",
    itemCol="anime_id",
    ratingCol="rating",
    rank=6,
    regParam=0.15,
    maxIter=8,
    seed=42,
    numUserBlocks=2,
    numItemBlocks=2,
    checkpointInterval=-1,
    coldStartStrategy="drop"
)

model = als.fit(train)

# 获取用户与动漫特征向量
user_factors = {
    int(row["id"]): np.array(row["features"])
    for row in model.userFactors.collect()
}

item_factors = {
    int(row["id"]): np.array(row["features"])
    for row in model.itemFactors.collect()
}

# 5. 仅使用训练集计算动漫热度
item_counts = train_pd.groupby("anime_id").size()

eligible_items = sorted([
    int(item)
    for item, count in item_counts.items()
    if count >= MIN_RATINGS and int(item) in item_factors
])

if len(eligible_items) < TOP_K:
    raise RuntimeError("可推荐动漫数量不足")

# 热门度采用 log(1 + 评分记录数)
pop_raw = np.log1p(
    item_counts.reindex(eligible_items).to_numpy()
)

# 归一化到 0～1
pop_range = pop_raw.max() - pop_raw.min()
pop_scores = (
    (pop_raw - pop_raw.min()) / pop_range
    if pop_range > 0
    else np.zeros_like(pop_raw)
)

item_matrix = np.stack([
    item_factors[item] for item in eligible_items
])

item_array = np.array(eligible_items)

# 6. 获取训练集最活跃的100名用户
user_counts = (
    train_pd.groupby("user_id").size()
    .reset_index(name="count")
    .sort_values(
        ["count", "user_id"],
        ascending=[False, True]
    )
)

selected_users = [
    int(u)
    for u in user_counts["user_id"]
    if int(u) in user_factors
][:USER_LIMIT]

# 7. 收集训练历史
seen = defaultdict(set)

for row in train_pd.itertuples(index=False):
    seen[int(row.user_id)].add(int(row.anime_id))

# 8. 验证集评分 >=8 的动漫视为相关
relevant = defaultdict(set)
eligible_set = set(eligible_items)

for row in valid_pd.itertuples(index=False):
    if (
        row.rating >= POSITIVE_THRESHOLD
        and int(row.anime_id) in eligible_set
    ):
        relevant[int(row.user_id)].add(int(row.anime_id))

# 9. 评价指标
def evaluate(top_items, truth):
    hits = sum(item in truth for item in top_items)

    precision = hits / TOP_K
    recall = hits / len(truth)
    hitrate = float(hits > 0)

    dcg = sum(
        1 / math.log2(i + 2)
        for i, item in enumerate(top_items)
        if item in truth
    )

    ideal = sum(
        1 / math.log2(i + 2)
        for i in range(min(len(truth), TOP_K))
    )

    ndcg = dcg / ideal if ideal else 0.0

    return precision, recall, hitrate, ndcg

# 10. 混合权重实验
print("\n===== 开始混合推荐验证 =====")

records = []
evaluated_users = 0

for uid in selected_users:
    truth = relevant[uid] - seen[uid]

    if not truth:
        continue

    # 排除用户在训练集中已评分的动漫
    mask = np.array([
        item not in seen[uid]
        for item in eligible_items
    ])

    candidates = item_array[mask]

    if len(candidates) < TOP_K:
        continue

    # ALS 预测分数：用户向量与动漫向量点积
    als_raw = item_matrix[mask] @ user_factors[uid]

    # 按当前用户的候选分数归一化
    score_range = als_raw.max() - als_raw.min()

    als_norm = (
        (als_raw - als_raw.min()) / score_range
        if score_range > 0
        else np.zeros_like(als_raw)
    )

    pop_norm = pop_scores[mask]

    for alpha in WEIGHTS:
        hybrid_scores = (
            alpha * als_norm
            + (1 - alpha) * pop_norm
        )

        # 按分数降序，动漫 ID 升序打破平局
        order = np.lexsort(
            (candidates, -hybrid_scores)
        )

        top_items = candidates[order[:TOP_K]].tolist()

        p, r, h, n = evaluate(top_items, truth)

        records.append({
            "user_id": uid,
            "alpha": alpha,
            "precision_at_10": p,
            "recall_at_10": r,
            "hitrate_at_10": h,
            "ndcg_at_10": n
        })

    evaluated_users += 1

if not records:
    raise RuntimeError("没有满足评估条件的用户")

# 11. 汇总各组参数
result = pd.DataFrame(records)

summary = (
    result.groupby("alpha", as_index=False)[[
        "precision_at_10",
        "recall_at_10",
        "hitrate_at_10",
        "ndcg_at_10"
    ]]
    .mean()
)

# 使用验证集 NDCG@10 选择混合权重
best = summary.sort_values(
    ["ndcg_at_10", "alpha"],
    ascending=[False, True]
).iloc[0]

summary.to_csv(
    "output/hybrid_validation.csv",
    index=False
)

result.to_csv(
    "output/hybrid_validation_users.csv",
    index=False
)

metadata = {
    "best_alpha": float(best["alpha"]),
    "evaluated_users": evaluated_users,
    "candidate_items": len(eligible_items),
    "train_count": len(train_pd),
    "validation_count": len(valid_pd),
    "test_count": int((buckets >= 85).sum()),
    "top_k": TOP_K,
    "positive_threshold": POSITIVE_THRESHOLD
}

with open(
    "output/hybrid_validation_meta.json",
    "w"
) as f:
    json.dump(metadata, f, indent=2)

print("\n===== 混合推荐验证结果 =====")
print(summary.to_string(index=False))

print("\n===== 验证集最佳权重 =====")
print(f"alpha = {best['alpha']:.2f}")
print(f"NDCG@10 = {best['ndcg_at_10']:.4f}")
print("参与评估用户：", evaluated_users)

print("\n验证完成，测试集尚未参与模型选择。")

spark.stop()
