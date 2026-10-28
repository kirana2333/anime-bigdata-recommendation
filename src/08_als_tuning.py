import csv
import json
from pathlib import Path

from pyspark.sql import SparkSession, functions as F
from pyspark.ml.recommendation import ALS
from pyspark.ml.evaluation import RegressionEvaluator

spark = (
    SparkSession.builder
    .appName("AnimeALSParameterTuning")
    .master("local[2]")
    .config("spark.driver.memory", "1g")
    .config("spark.sql.shuffle.partitions", "4")
    .config("spark.default.parallelism", "2")
    .config("spark.ui.enabled", "false")
    .getOrCreate()
)
spark.sparkContext.setLogLevel("ERROR")

# 读取之前清洗好的评分数据
ratings = (
    spark.read
    .option("header", True)
    .schema("user_id INT, anime_id INT, rating FLOAT")
    .csv("data/processed/ratings_als_sample.csv")
    .na.drop()
)

# 重新划分数据：训练集、验证集、测试集
train, valid, test = ratings.randomSplit(
    [0.70, 0.15, 0.15], seed=42
)

train = train.cache()
valid = valid.cache()
test = test.cache()

train_count = train.count()
valid_count = valid.count()
test_count = test.count()

print("===== 数据集划分 =====")
print("训练集：", train_count)
print("验证集：", valid_count)
print("测试集：", test_count)

# 四组参数
configs = [
    (6, 0.15),
    (8, 0.15),
    (8, 0.30),
    (12, 0.15)
]

rmse_eval = RegressionEvaluator(
    labelCol="rating",
    predictionCol="prediction",
    metricName="rmse"
)

mae_eval = RegressionEvaluator(
    labelCol="rating",
    predictionCol="prediction",
    metricName="mae"
)

def make_als(rank, reg):
    return ALS(
        userCol="user_id",
        itemCol="anime_id",
        ratingCol="rating",
        rank=rank,
        maxIter=8,
        regParam=reg,
        seed=42,
        coldStartStrategy="drop",
        numUserBlocks=2,
        numItemBlocks=2,
        checkpointInterval=-1
    )

results = []

# 逐组训练并在验证集上比较
for idx, (rank, reg) in enumerate(configs, 1):
    print(f"\n===== 模型 {idx}/4 =====")
    print(f"rank={rank}, regParam={reg}")

    model = make_als(rank, reg).fit(train)

    predictions = model.transform(valid).cache()
    evaluated = predictions.count()

    if evaluated == 0:
        raise RuntimeError("验证集无有效预测记录")

    rmse = rmse_eval.evaluate(predictions)
    mae = mae_eval.evaluate(predictions)
    coverage = evaluated / valid_count

    results.append({
        "rank": rank,
        "regParam": reg,
        "RMSE": float(rmse),
        "MAE": float(mae),
        "coverage": float(coverage)
    })

    print(f"验证集 RMSE: {rmse:.4f}")
    print(f"验证集 MAE: {mae:.4f}")
    print(f"预测覆盖率: {coverage:.2%}")

    predictions.unpersist()

# 保存参数比较结果
Path("output").mkdir(exist_ok=True)

with open(
    "output/als_tuning_results.csv",
    "w",
    newline=""
) as f:
    writer = csv.DictWriter(
        f,
        fieldnames=[
            "rank", "regParam", "RMSE",
            "MAE", "coverage"
        ]
    )
    writer.writeheader()
    writer.writerows(results)

# 根据验证集 RMSE 选择最佳参数
best = min(results, key=lambda x: x["RMSE"])

print("\n===== 参数对比结果 =====")
for row in results:
    print(
        f"rank={row['rank']:2d}, "
        f"reg={row['regParam']:.2f}, "
        f"RMSE={row['RMSE']:.4f}, "
        f"MAE={row['MAE']:.4f}, "
        f"coverage={row['coverage']:.2%}"
    )

print("\n===== 最优参数 =====")
print("rank:", best["rank"])
print("regParam:", best["regParam"])

# 用训练集 + 验证集重新训练最佳模型
final_train = train.unionByName(valid).cache()
final_train_count = final_train.count()

print("\n===== 训练最终模型 =====")
final_model = make_als(
    best["rank"], best["regParam"]
).fit(final_train)

# 只在最终阶段使用测试集
test_predictions = final_model.transform(test).cache()
test_evaluated = test_predictions.count()

if test_evaluated == 0:
    raise RuntimeError("测试集无有效预测记录")

test_rmse = rmse_eval.evaluate(test_predictions)
test_mae = mae_eval.evaluate(test_predictions)
test_coverage = test_evaluated / test_count

# 简单基线：始终预测训练集平均评分
mean_rating = final_train.agg(
    F.avg("rating")
).first()[0]

baseline_predictions = test_predictions.withColumn(
    "prediction",
    F.lit(float(mean_rating))
)

baseline_rmse = rmse_eval.evaluate(baseline_predictions)

print("\n===== 最终测试集评估 =====")
print(f"ALS RMSE: {test_rmse:.4f}")
print(f"ALS MAE: {test_mae:.4f}")
print(f"基线 RMSE: {baseline_rmse:.4f}")
print(f"预测覆盖率: {test_coverage:.2%}")

# 保存最佳模型，不覆盖之前的模型
Path("models").mkdir(exist_ok=True)

final_model.write().overwrite().save(
    "models/als_tuned"
)

summary = {
    "best_rank": int(best["rank"]),
    "best_regParam": float(best["regParam"]),
    "train_count": int(train_count),
    "valid_count": int(valid_count),
    "test_count": int(test_count),
    "final_train_count": int(final_train_count),
    "test_evaluated": int(test_evaluated),
    "test_rmse": float(test_rmse),
    "test_mae": float(test_mae),
    "baseline_rmse": float(baseline_rmse),
    "test_coverage": float(test_coverage)
}

with open(
    "output/als_tuning_summary.json",
    "w"
) as f:
    json.dump(summary, f, indent=2)

print("\n===== 实验完成 =====")
print("模型：models/als_tuned")
print("对比结果：output/als_tuning_results.csv")
print("测试结果：output/als_tuning_summary.json")

test_predictions.unpersist()
final_train.unpersist()
train.unpersist()
valid.unpersist()
test.unpersist()

spark.stop()
