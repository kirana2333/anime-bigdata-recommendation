import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path

import pandas as pd
from pyspark.sql import SparkSession
from pyspark.ml.recommendation import ALS

TOP_K = 10
USER_LIMIT = 100
MIN_ITEM_RATINGS = 20
POSITIVE_RATING = 8

Path("output").mkdir(exist_ok=True)

# 1. 加载评分数据
ratings = pd.read_csv(
    "data/processed/ratings_als_sample.csv"
)

# 2. 使用稳定哈希划分训练集和测试集
# 相同用户-动漫组合始终分配到相同集合
def split_bucket(user_id, anime_id):
    key = f"ranking-v1:{int(user_id)}:{int(anime_id)}"
    digest = hashlib.sha256(key.encode()).digest()
    return int.from_bytes(digest[:8], "big") % 100

buckets = [
    split_bucket(u, a)
    for u, a in zip(
        ratings["user_id"],
        ratings["anime_id"]
    )
]

train_pd = ratings[
    pd.Series(buckets, index=ratings.index) < 85
].copy()

test_pd = ratings[
    pd.Series(buckets, index=ratings.index) >= 85
].copy()

print("===== 数据划分 =====")
print("训练记录：", len(train_pd))
print("测试记录：", len(test_pd))

# 3. 启动 Spark
spark = (
    SparkSession.builder
    .appName("AnimeRankingEvaluation")
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

# 4. 使用此前选出的参数训练
print("===== 训练排序评估专用 ALS 模型 =====")

als = ALS(
    userCol="user_id",
    itemCol="anime_id",
    ratingCol="rating",
    rank=6,
    regParam=0.15,
    maxIter=8,
    seed=42,
    coldStartStrategy="drop",
    numUserBlocks=2,
    numItemBlocks=2,
    checkpointInterval=-1
)

model = als.fit(train)

print("ALS 训练完成")

# 5. 只使用训练集统计动漫热度
item_counts = (
    train_pd.groupby("anime_id")
    .size()
    .sort_values(ascending=False)
)

known_items = {
    int(row["id"])
    for row in model.itemFactors.select("id").collect()
}

eligible_items = [
    int(item_id)
    for item_id, count in item_counts.items()
    if count >= MIN_ITEM_RATINGS
    and int(item_id) in known_items
]

eligible_set = set(eligible_items)

# 6. 按训练集活跃程度选择100名用户
known_users = {
    int(row["id"])
    for row in model.userFactors.select("id").collect()
}

user_counts = (
    train_pd.groupby("user_id")
    .size()
    .sort_values(ascending=False)
)

selected_users = [
    int(u)
    for u in user_counts.index
    if int(u) in known_users
][:USER_LIMIT]

# 7. 用户在训练集中已有交互的动漫
seen = {
    u: set(
        train_pd.loc[
            train_pd["user_id"] == u, "anime_id"
        ].astype(int)
    )
    for u in selected_users
}

# 8. 测试集里评分 >=8 的作品作为相关作品
positive_test = test_pd[
    (test_pd["rating"] >= POSITIVE_RATING)
    & (test_pd["anime_id"].isin(eligible_set))
]

relevant = defaultdict(set)

for row in positive_test.itertuples(index=False):
    relevant[int(row.user_id)].add(int(row.anime_id))

# 只评估具有至少一条可推荐测试集正例的用户
eval_users = [
    u for u in selected_users
    if len(relevant[u] - seen[u]) > 0
]

if not eval_users:
    raise RuntimeError("没有满足评估条件的用户")

# 9. 构造候选集合，排除训练集中已评分作品
candidates = [
    (u, item)
    for u in eval_users
    for item in eligible_items
    if item not in seen[u]
]

candidate_df = spark.createDataFrame(
    candidates,
    "user_id INT, anime_id INT"
)

print("===== 预测候选动漫 =====")
print("参与评估的用户：", len(eval_users))
print("候选动漫数量：", len(eligible_items))
print("待预测组合：", len(candidates))

# 10. ALS 批量预测
predictions = model.transform(candidate_df).select(
    "user_id", "anime_id", "prediction"
).collect()

als_scores = defaultdict(list)

for row in predictions:
    score = float(row["prediction"])

    if math.isfinite(score):
        als_scores[int(row["user_id"])].append(
            (int(row["anime_id"]), score)
        )

# 11. 评价函数
def evaluate_ranked(items, truth):
    top = items[:TOP_K]

    hits = sum(item in truth for item in top)

    precision = hits / TOP_K
    recall = hits / len(truth)
    hitrate = float(hits > 0)

    dcg = sum(
        1 / math.log2(i + 2)
        for i, item in enumerate(top)
        if item in truth
    )

    ideal = sum(
        1 / math.log2(i + 2)
        for i in range(min(len(truth), TOP_K))
    )

    ndcg = dcg / ideal if ideal else 0.0

    return precision, recall, hitrate, ndcg

# 12. ALS 与热门动漫推荐对比
records = []

for u in eval_users:
    truth = relevant[u] - seen[u]

    als_top = [
        item
        for item, score in sorted(
            als_scores[u],
            key=lambda x: (-x[1], x[0])
        )[:TOP_K]
    ]

    # 简单基线：按训练集评分记录数量排序
    popular_top = [
        item for item in eligible_items
        if item not in seen[u]
    ][:TOP_K]

    for name, ranked in [
        ("ALS", als_top),
        ("Popularity", popular_top)
    ]:
        p, r, h, n = evaluate_ranked(ranked, truth)

        records.append({
            "user_id": u,
            "model": name,
            "relevant_count": len(truth),
            "precision_at_10": p,
            "recall_at_10": r,
            "hitrate_at_10": h,
            "ndcg_at_10": n
        })

# 13. 保存结果
result = pd.DataFrame(records)

summary = (
    result.groupby("model")[
        [
            "precision_at_10",
            "recall_at_10",
            "hitrate_at_10",
            "ndcg_at_10"
        ]
    ]
    .mean()
)

result.to_csv(
    "output/ranking_user_results.csv",
    index=False
)

summary.to_csv(
    "output/ranking_evaluation.csv"
)

metadata = {
    "evaluated_users": len(eval_users),
    "selected_users": len(selected_users),
    "train_count": len(train_pd),
    "test_count": len(test_pd),
    "min_item_ratings": MIN_ITEM_RATINGS,
    "positive_rating_threshold": POSITIVE_RATING,
    "top_k": TOP_K
}

with open(
    "output/ranking_evaluation_meta.json",
    "w"
) as f:
    json.dump(metadata, f, indent=2)

print("\n===== 推荐排序评估结果 =====")
print(summary.to_string())

print("\n评估完成，结果已保存到 output 目录")

spark.stop()
