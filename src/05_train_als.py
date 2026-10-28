import json
from pathlib import Path

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.ml.recommendation import ALS
from pyspark.ml.evaluation import RegressionEvaluator

# 1. 创建 Spark 环境
spark = (
    SparkSession.builder
    .appName("AnimeALS")
    .master("local[2]")
    .config("spark.driver.memory", "1g")
    .config("spark.sql.shuffle.partitions", "4")
    .config("spark.default.parallelism", "2")
    .config("spark.ui.enabled", "false")
    .getOrCreate()
)

spark.sparkContext.setLogLevel("ERROR")

# 2. 读取预处理后的评分数据
ratings = (
    spark.read
    .option("header", True)
    .schema("user_id INT, anime_id INT, rating FLOAT")
    .csv("data/processed/ratings_als_sample.csv")
    .na.drop()
)

print("===== 数据加载完成 =====")
print("评分记录：", ratings.count())

# 3. 划分训练集与测试集
train, test = ratings.randomSplit(
    [0.8, 0.2],
    seed=42
)

train = train.cache()
test = test.cache()

train_count = train.count()
test_count = test.count()

print("训练集：", train_count)
print("测试集：", test_count)

# 4. 配置 ALS 推荐模型
als = ALS(
    userCol="user_id",
    itemCol="anime_id",
    ratingCol="rating",
    rank=8,
    maxIter=8,
    regParam=0.15,
    coldStartStrategy="drop",
    nonnegative=False,
    numUserBlocks=2,
    numItemBlocks=2,
    checkpointInterval=-1,
    seed=42
)

print("===== 开始训练 ALS 模型 =====")

model = als.fit(train)

print("===== 模型训练完成 =====")

# 5. 在测试集上预测评分
predictions = model.transform(test).cache()

evaluated_count = predictions.count()

if evaluated_count == 0:
    raise RuntimeError("测试集没有可评估的预测记录")

# 6. 计算 ALS 预测误差
rmse_evaluator = RegressionEvaluator(
    labelCol="rating",
    predictionCol="prediction",
    metricName="rmse"
)

mae_evaluator = RegressionEvaluator(
    labelCol="rating",
    predictionCol="prediction",
    metricName="mae"
)

rmse = rmse_evaluator.evaluate(predictions)
mae = mae_evaluator.evaluate(predictions)

# 7. 与平均评分基线比较
mean_rating = train.agg(
    F.avg("rating")
).first()[0]

baseline = predictions.withColumn(
    "baseline_prediction",
    F.lit(mean_rating)
)

baseline_rmse = RegressionEvaluator(
    labelCol="rating",
    predictionCol="baseline_prediction",
    metricName="rmse"
).evaluate(baseline)

print("===== 模型评估结果 =====")
print(f"ALS RMSE: {rmse:.4f}")
print(f"ALS MAE: {mae:.4f}")
print(f"基线 RMSE: {baseline_rmse:.4f}")
print(f"测试集预测覆盖率: {evaluated_count/test_count:.2%}")

# 8. 保存模型
Path("models").mkdir(exist_ok=True)
model.write().overwrite().save("models/als_v1")

# 9. 保存评估结果
Path("output").mkdir(exist_ok=True)

metrics = {
    "train_count": train_count,
    "test_count": test_count,
    "evaluated_count": evaluated_count,
    "rmse": rmse,
    "mae": mae,
    "baseline_rmse": baseline_rmse
}

with open("output/als_metrics.json", "w") as f:
    json.dump(metrics, f, indent=2)

print("模型已保存到 models/als_v1")
print("评估结果已保存到 output/als_metrics.json")

spark.stop()
