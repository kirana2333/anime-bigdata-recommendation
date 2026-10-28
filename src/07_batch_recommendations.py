import html
import numpy as np
import pandas as pd
from pathlib import Path

from pyspark.sql import SparkSession
from pyspark.ml.recommendation import ALSModel

# 每次生成20名用户的推荐
USER_LIMIT = 20
TOP_K = 10
MIN_RATINGS = 20

spark = (
    SparkSession.builder
    .appName("AnimeBatchRecommendation")
    .master("local[2]")
    .config("spark.driver.memory", "1g")
    .config("spark.sql.shuffle.partitions", "4")
    .config("spark.ui.enabled", "false")
    .getOrCreate()
)
spark.sparkContext.setLogLevel("ERROR")

# 1. 加载已训练好的 ALS 模型
model = ALSModel.load("models/als_tuned")

# 2. 读取已经清洗好的评分数据
ratings = pd.read_csv(
    "data/processed/ratings_als_sample.csv"
)

# 3. 获取模型已经学习的用户与动漫 ID
known_users = {
    int(row["id"])
    for row in model.userFactors.select("id").collect()
}

known_items = {
    int(row["id"])
    for row in model.itemFactors.select("id").collect()
}

# 4. 选取历史评分较多的20名用户
user_counts = (
    ratings.groupby("user_id")
    .size()
    .reset_index(name="history_count")
)

selected_users = (
    user_counts[user_counts["user_id"].isin(known_users)]
    .nlargest(USER_LIMIT, "history_count")
)

user_ids = selected_users["user_id"].astype(int).tolist()

print("===== 选中的用户 =====")
print("用户数量：", len(user_ids))
print("用户 ID：", user_ids)

# 5. 过滤样本评分数量不足20条的动漫
item_counts = ratings.groupby("anime_id").size()

eligible_items = [
    int(item_id)
    for item_id, count in item_counts.items()
    if count >= MIN_RATINGS and int(item_id) in known_items
]

print("符合要求的动漫数量：", len(eligible_items))

# 6. 记录每名用户已经评分的动漫
seen = {
    user_id: set(
        ratings.loc[
            ratings["user_id"] == user_id, "anime_id"
        ].astype(int)
    )
    for user_id in user_ids
}

# 7. 为每名用户生成未评分的候选动漫
candidate_rows = [
    (user_id, anime_id)
    for user_id in user_ids
    for anime_id in eligible_items
    if anime_id not in seen[user_id]
]

if not candidate_rows:
    raise RuntimeError("没有可推荐的候选动漫")

candidates = spark.createDataFrame(
    candidate_rows,
    "user_id INT, anime_id INT"
)

print("待预测的用户-动漫组合：", len(candidate_rows))

# 8. 批量预测偏好分数
rows = (
    model.transform(candidates)
    .select("user_id", "anime_id", "prediction")
    .collect()
)

predictions = pd.DataFrame(
    [
        (row["user_id"], row["anime_id"], row["prediction"])
        for row in rows
        if row["prediction"] is not None
        and np.isfinite(row["prediction"])
    ],
    columns=["user_id", "anime_id", "predicted_rating"]
)

# 9. 关联动漫基本信息
anime = pd.read_csv("data/raw/anime.csv")

anime["name"] = anime["name"].fillna("").map(html.unescape)
anime["members"] = pd.to_numeric(
    anime["members"], errors="coerce"
).fillna(0)

result = predictions.merge(
    anime,
    on="anime_id",
    how="inner"
)

# 10. 每名用户分别选取TOP 10
result = result.sort_values(
    ["user_id", "predicted_rating", "members"],
    ascending=[True, False, False]
)

top10 = (
    result.groupby("user_id", sort=False)
    .head(TOP_K)
    .copy()
)

top10["rank"] = top10.groupby("user_id").cumcount() + 1

top10 = top10.rename(columns={
    "rating": "dataset_rating"
})

# 11. 保存所有用户的推荐结果
Path("output").mkdir(exist_ok=True)

columns = [
    "user_id", "rank", "anime_id", "name",
    "type", "genre", "predicted_rating",
    "dataset_rating", "members"
]

top10[columns].to_csv(
    "output/recommendations_multiuser.csv",
    index=False
)

print("\n===== 批量推荐完成 =====")
print("成功生成推荐的用户：", top10["user_id"].nunique())
print("推荐结果总数：", len(top10))
print("输出文件：output/recommendations_multiuser.csv")

# 12. 展示其中一名用户的推荐结果
display_user = 478 if 478 in user_ids else user_ids[0]

print(f"\n===== 用户 {display_user} 的 TOP 10 =====")

print(
    top10[top10["user_id"] == display_user][
        ["rank", "name", "predicted_rating"]
    ].to_string(index=False)
)

spark.stop()
