from pyspark.sql import SparkSession

# 1. 创建 Spark 运行环境
spark = (
    SparkSession.builder
    .appName("AnimeRanking")
    .master("local[2]")
    .config("spark.driver.memory", "1g")
    .config("spark.ui.enabled", "false")
    .getOrCreate()
)

spark.sparkContext.setLogLevel("ERROR")

# 2. 从 HDFS 读取动漫数据
path = "hdfs://172.18.0.2:9000/user/bigdata/anime/raw/anime.csv"

df = (
    spark.read
    .option("header", True)
    .option("inferSchema", True)
    .csv(path)
)

# 3. 注册 SQL 临时视图
df.createOrReplaceTempView("anime")

# 4. 使用 Spark SQL 查询评分排行榜
result = spark.sql("""
    SELECT
        name,
        rating,
        members
    FROM anime
    WHERE rating IS NOT NULL
      AND type = 'TV'
      AND members >= 10000
    ORDER BY rating DESC, members DESC
    LIMIT 10
""")

# 5. 显示排行榜
print("===== TV 动画评分 TOP 10 =====")
result.show(10, truncate=False)

spark.stop()
