import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from pyspark.sql import SparkSession
from pyspark.ml.recommendation import ALS

# ===== 实验设置 =====
TOP_K = 10
USER_LIMIT = 100
MIN_RATINGS = 20
POSITIVE_THRESHOLD = 8

Path("output").mkdir(exist_ok=True)
Path("models").mkdir(exist_ok=True)

# 1. 读取验证阶段选出的最佳权重
with open("output/hybrid_validation_meta.json") as f:
    validation_meta = json.load(f)

BEST_ALPHA = float(validation_meta["best_alpha"])

if not 0 <= BEST_ALPHA <= 1:
    raise ValueError("混合权重必须在0到1之间")

print("===== 最终测试配置 =====")
print("最佳混合权重：", BEST_ALPHA)
print("推荐数量：", TOP_K)

# 2. 读取评分数据
ratings = pd.read_csv(
    "data/processed/ratings_als_sample.csv"
)

# 必须与之前的验证实验使用相同划分规则
def get_bucket(user_id, anime_id):
    key = f"hybrid-v1:{int(user_id)}:{int(anime_id)}"
    digest = hashlib.sha256(key.encode()).digest()
    return int.from_bytes(digest[:8], "big") % 100

buckets = np.array([
    get_bucket(u, a)
    for u, a in zip(
        ratings["user_id"],
        ratings["anime_id"]
    )
])

# 训练集 + 验证集
train_pd = ratings.loc[buckets < 85].copy()

# 最终独立测试集
test_pd = ratings.loc[buckets >= 85].copy()

print("\n===== 数据划分 =====")
print("最终训练集：", len(train_pd))
print("独立测试集：", len(test_pd))

# 3. 初始化 Spark
spark = (
    SparkSession.builder
    .appName("AnimeHybridFinalTest")
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

# 4. 使用验证阶段确定的参数重新训练
print("\n===== 训练最终 ALS 模型 =====")

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

print("最终 ALS 模型训练完成")

# 5. 提取用户和动漫特征
user_factors = {
    int(row["id"]): np.array(row["features"])
    for row in model.userFactors.collect()
}

item_factors = {
    int(row["id"]): np.array(row["features"])
    for row in model.itemFactors.collect()
}

# 6. 根据训练集构造候选动漫
item_counts = train_pd.groupby("anime_id").size()

eligible_items = sorted([
    int(item)
    for item, count in item_counts.items()
    if count >= MIN_RATINGS
    and int(item) in item_factors
])

if len(eligible_items) < TOP_K:
    raise RuntimeError("候选动漫数量不足")

eligible_set = set(eligible_items)
item_array = np.array(eligible_items)

item_matrix = np.stack([
    item_factors[item]
    for item in eligible_items
])

# 7. 计算热门度分数
pop_raw = np.log1p(
    item_counts.reindex(eligible_items).to_numpy(
        dtype=float
    )
)

def normalize(values):
    low = values.min()
    high = values.max()

    if high <= low:
        return np.zeros_like(values, dtype=float)

    return (values - low) / (high - low)

pop_scores = normalize(pop_raw)

# 8. 选择训练集中最活跃的100名用户
user_counts = (
    train_pd.groupby("user_id")
    .size()
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

# 9. 统计已评分作品
seen = defaultdict(set)

for row in train_pd.itertuples(index=False):
    seen[int(row.user_id)].add(int(row.anime_id))

# 10. 构建测试集正例
relevant = defaultdict(set)

for row in test_pd.itertuples(index=False):
    if (
        row.rating >= POSITIVE_THRESHOLD
        and int(row.anime_id) in eligible_set
    ):
        relevant[int(row.user_id)].add(
            int(row.anime_id)
        )

# 11. 定义评价指标
def evaluate(top_items, truth):
    hits = sum(
        item in truth for item in top_items
    )

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

# 12. 对三种推荐策略进行同条件测试
print("\n===== 开始独立测试集评估 =====")

records = []
evaluated_users = 0

for uid in selected_users:

    truth = relevant[uid] - seen[uid]

    if not truth:
        continue

    mask = np.array([
        item not in seen[uid]
        for item in eligible_items
    ])

    candidates = item_array[mask]

    if len(candidates) < TOP_K:
        continue

    # ALS 预测分数
    als_raw = (
        item_matrix[mask] @ user_factors[uid]
    )

    als_norm = normalize(als_raw)

    # 热门度分数
    pop_norm = pop_scores[mask]

    strategies = {
        "ALS": als_norm,
        "Popularity": pop_norm,
        "Hybrid": (
            BEST_ALPHA * als_norm
            + (1 - BEST_ALPHA) * pop_norm
        )
    }

    for name, scores in strategies.items():

        # 分数降序，ID升序打破平局
        order = np.lexsort(
            (candidates, -scores)
        )

        top_items = candidates[
            order[:TOP_K]
        ].tolist()

        p, r, h, n = evaluate(
            top_items, truth
        )

        records.append({
            "user_id": uid,
            "model": name,
            "precision_at_10": p,
            "recall_at_10": r,
            "hitrate_at_10": h,
            "ndcg_at_10": n
        })

    evaluated_users += 1

if not records:
    raise RuntimeError("没有满足测试条件的用户")

# 13. 汇总实验结果
result = pd.DataFrame(records)

summary = (
    result.groupby("model", sort=False)[[
        "precision_at_10",
        "recall_at_10",
        "hitrate_at_10",
        "ndcg_at_10"
    ]]
    .mean()
    .reset_index()
)

print("\n===== 最终测试集评估结果 =====")
print(summary.to_string(index=False))

print("\n参与测试的用户：", evaluated_users)
print("候选动漫数量：", len(eligible_items))

# 14. 保存实验结果
result.to_csv(
    "output/hybrid_final_users.csv",
    index=False
)

summary.to_csv(
    "output/hybrid_final_results.csv",
    index=False
)

metadata = {
    "best_alpha": BEST_ALPHA,
    "train_count": len(train_pd),
    "test_count": len(test_pd),
    "evaluated_users": evaluated_users,
    "candidate_items": len(eligible_items),
    "top_k": TOP_K,
    "positive_threshold": POSITIVE_THRESHOLD,
    "min_item_ratings": MIN_RATINGS
}

with open(
    "output/hybrid_final_meta.json",
    "w"
) as f:
    json.dump(metadata, f, indent=2)

# 保存训练好的最终模型
model.write().overwrite().save(
    "models/als_hybrid_final"
)

# 保存候选动漫及热门度信息
pd.DataFrame({
    "anime_id": eligible_items,
    "rating_count": item_counts.reindex(
        eligible_items
    ).to_numpy(),
    "popularity_score": pop_scores
}).to_csv(
    "output/hybrid_item_popularity.csv",
    index=False
)

print("\n===== 实验完成 =====")
print("最终模型：models/als_hybrid_final")
print("测试结果：output/hybrid_final_results.csv")
print("候选动漫：output/hybrid_item_popularity.csv")

spark.stop()
