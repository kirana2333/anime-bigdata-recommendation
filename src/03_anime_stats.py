from pyspark.sql import SparkSession
from pyspark.sql import functions as F

spark = (
    SparkSession.builder
    .appName("AnimeStatistics")
    .master("local[2]")
    .config("spark.driver.memory", "1g")
    .config("spark.ui.enabled", "false")
    .getOrCreate()
)

spark.sparkContext.setLogLevel("ERROR")

# 1. 从 HDFS 读取原始数据
path = "hdfs://172.18.0.2:9000/user/bigdata/anime/raw/anime.csv"

df = spark.read.csv(
    path,
    header=True,
    inferSchema=True
)

# 2. 数据清洗
# 排除缺失评分和不合理评分
clean = df.filter(
    F.col("rating").between(0, 10)
)

print("===== 原始动漫数量 =====")
print(df.count())

print("===== 清洗后动漫数量 =====")
print(clean.count())

# 3. 按动画形式统计
type_stats = (
    clean
    .filter(F.col("type").isNotNull())
    .groupBy("type")
    .agg(
        F.count("*").alias("anime_count"),
        F.round(F.avg("rating"), 2).alias("avg_rating")
    )
    .orderBy(F.desc("anime_count"))
)

print("===== 动画形式统计 =====")
type_stats.show(truncate=False)

# 4. 将一部动漫的多个题材拆开
genre_df = (
    clean
    .filter(F.col("genre").isNotNull())
    .withColumn(
        "single_genre",
        F.explode(F.split(F.col("genre"), ","))
    )
    .withColumn(
        "single_genre",
        F.trim(F.col("single_genre"))
    )
    .filter(F.col("single_genre") != "")
)

# 5. 统计出现次数最多的15个题材
genre_stats = (
    genre_df
    .groupBy("single_genre")
    .agg(F.count("*").alias("anime_count"))
    .orderBy(F.desc("anime_count"))
    .limit(15)
)

print("===== 热门动漫题材 TOP 15 =====")
genre_stats.show(15, truncate=False)

# 6. 导出分析结果，供网页使用
type_stats.coalesce(1).write.mode("overwrite").option(
    "header", True
).csv("output/type_stats")

genre_stats.coalesce(1).write.mode("overwrite").option(
    "header", True
).csv("output/genre_stats")

spark.stop()

print("统计完成！结果已保存到 output 目录。")
