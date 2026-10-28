from pyspark.sql import SparkSession

# 1. 创建 SparkSession
spark = (
    SparkSession.builder
    .appName("AnimeAnalysis")
    .master("local[2]")
    .config("spark.driver.memory", "1g")
    .config("spark.ui.enabled", "false")
    .config("spark.hadoop.dfs.client.use.datanode.hostname", "false")
    .getOrCreate()
)

spark.sparkContext.setLogLevel("ERROR")

# 2. 读取 HDFS 中的动漫数据
hdfs_path = "hdfs://172.18.0.2:9000/user/bigdata/anime/raw/anime.csv"

df = (
    spark.read
    .option("header", True)
    .option("inferSchema", True)
    .csv(hdfs_path)
)

# 3. 查看数据结构
print("===== 数据字段 =====")
df.printSchema()

# 4. 统计动漫总数
print("===== 动漫总数 =====")
print(df.count())

# 5. 查看前五条数据
print("===== 前五条动漫数据 =====")
df.select("anime_id", "name", "rating", "members").show(
    5, truncate=False
)

spark.stop()
