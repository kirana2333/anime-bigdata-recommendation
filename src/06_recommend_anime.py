import csv
import html
from pathlib import Path

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.ml.recommendation import ALSModel

spark = (
    SparkSession.builder
    .appName("AnimeRecommendation")
    .master("local[2]")
    .config("spark.driver.memory", "1g")
    .config("spark.sql.shuffle.partitions", "4")
    .config("spark.ui.enabled", "false")
    .getOrCreate()
)

spark.sparkContext.setLogLevel("ERROR")

# 1. 加载之前训练好的 ALS 模型
model = ALSModel.load("models/als_v1")

# 2. 读取用户评分历史
ratings = (
    spark.read
    .option("header", True)
    .schema("user_id INT, anime_id INT, rating FLOAT")
    .csv("data/processed/ratings_als_sample.csv")
)

# 3. 从模型已学习的用户中选择评分最多的一位
known_users = model.userFactors.select(
    F.col("id").alias("user_id")
)

user_stats = (
    ratings.groupBy("user_id")
    .agg(F.count("*").alias("rating_count"))
    .join(known_users, "user_id")
    .orderBy(F.desc("rating_count"))
)

selected = user_stats.first()
user_id = selected["user_id"]

print("===== 当前推荐用户 =====")
print("用户 ID:", user_id)
print("历史评分数量:", selected["rating_count"])

# 4. 找出该用户已有评分的动漫
seen = (
    ratings
    .filter(F.col("user_id") == user_id)
    .select("anime_id")
    .distinct()
)

# 5. 从模型已学习的动漫中排除已评分作品
# 先提取模型已学习的动漫 ID，避免特征表重复关联
# 设置候选动漫的最低评分数量
min_rating_count = 20

# 统计每部动漫在样本中的评分记录数量
popular_ids = {
    int(row["anime_id"])
    for row in (
        ratings.groupBy("anime_id")
        .count()
        .filter(F.col("count") >= min_rating_count)
        .select("anime_id")
        .collect()
    )
}

# 仅保留 ALS 已学习且评分数量足够的动漫
item_ids = [
    int(row["id"])
    for row in model.itemFactors.select("id").collect()
    if int(row["id"]) in popular_ids
]

if not item_ids:
    raise RuntimeError("没有满足评分数量要求的动漫")

print("满足最低评分数量要求的动漫：", len(item_ids))

candidate_items = spark.createDataFrame(
    [(anime_id,) for anime_id in item_ids],
    "anime_id INT"
)

# 排除用户已经评分的动漫
unseen = candidate_items.join(
    seen, "anime_id", "left_anti"
)

# 6. 构建待预测的用户-动漫组合
user_df = spark.createDataFrame(
    [(int(user_id),)],
    "user_id INT"
)

candidates = user_df.crossJoin(unseen)

# 7. 预测该用户对每部候选动漫的评分
predictions = (
    model.transform(candidates)
    .filter(F.col("prediction").isNotNull())
)

# 8. 读取动漫名称等信息
anime = (
    spark.read
    .option("header", True)
    .option("inferSchema", True)
    .csv("data/raw/anime.csv")
    .select("anime_id", "name", "genre", "type", "rating")
)

# 9. 关联动漫信息，输出预测评分最高的10部
top10 = (
    predictions
    .join(anime, "anime_id")
    .orderBy(F.desc("prediction"))
    .limit(10)
    .collect()
)

print("\n===== ALS 个性化推荐 TOP 10 =====")

Path("output").mkdir(exist_ok=True)

with open(
    "output/recommendations_top10.csv",
    "w",
    newline="",
    encoding="utf-8"
) as f:
    writer = csv.writer(f)
    writer.writerow([
        "user_id", "anime_id", "name",
        "genre", "type", "predicted_rating"
    ])

    for i, row in enumerate(top10, 1):
        name = html.unescape(row["name"] or "")
        score = float(row["prediction"])

        print(f"{i}. {name} 预测评分: {score:.2f}")

        writer.writerow([
            user_id,
            row["anime_id"],
            name,
            row["genre"],
            row["type"],
            round(score, 4)
        ])

print("\n推荐结果已保存至 output/recommendations_top10.csv")

spark.stop()
