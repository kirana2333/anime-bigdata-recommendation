# Anime Data Lab
## 基于 Hadoop 与 Spark 的动漫数据分析及个性化推荐系统

本项目为大数据课程独立开发项目，使用 Hadoop HDFS、
Apache Spark、Spark MLlib 和 Streamlit，
实现动漫数据统计分析、交互式检索、个性化推荐及模型评估。

## 1. 项目功能

- 动漫数据清洗与统计分析
- 动画形式及热门题材可视化
- 动漫名称、类型、题材、评分联合检索
- 基于 Spark MLlib ALS 的协同过滤推荐
- 基于动漫热门度的推荐
- ALS 与热门度融合的 Hybrid 混合推荐
- 多用户推荐结果展示
- 模型参数对比与推荐质量评估

## 2. 技术架构

| 技术 | 主要用途 |
|---|---|
| Ubuntu 24.04 | Linux 实验环境 |
| Docker | Hadoop 集群容器管理 |
| Hadoop HDFS | 分布式数据存储 |
| Apache Spark SQL | 数据清洗与统计 |
| Spark MLlib ALS | 协同过滤模型训练 |
| Python / Pandas / NumPy | 数据处理与推荐融合 |
| Streamlit / Altair | Web 界面与数据可视化 |
| Git / GitHub | 代码管理与项目交付 |

Hadoop 环境采用三节点 Docker 集群。
Spark 主要以 local[2] 模式运行，以适应虚拟机资源限制。

原始数据曾上传至 HDFS，当前分析和模型训练程序
主要读取本地 CSV 数据文件。

## 3. 数据集

数据来源：

https://www.kaggle.com/datasets/CooperUnion/anime-recommendations-database

主要文件：

- anime.csv：动漫名称、形式、题材、评分及关注人数
- rating.csv：匿名用户 ID、动漫 ID 和用户评分

原始数据包含约 12,294 部动漫和 781 万条评分记录。

ALS 实验使用约 7.7 万条有效评分样本，
并过滤无效评分。

由于文件体积较大，GitHub 仓库不包含原始数据集。
请从上述数据来源下载文件并放入 data/raw/。

## 4. 推荐算法

### 4.1 ALS 协同过滤

使用 Spark MLlib ALS 学习用户与动漫的潜在特征，
根据用户历史评分预测偏好。

通过比较 rank 和 regParam 参数选择模型。

本次参数实验选出的配置：

- rank = 6
- regParam = 0.15
- maxIter = 8

独立测试集评分预测结果：

| 指标 | 结果 |
|---|---:|
| ALS RMSE | 1.2036 |
| ALS MAE | 0.9140 |
| 平均评分基线 RMSE | 1.5339 |
| 预测覆盖率 | 98.32% |

### 4.2 Hybrid 混合推荐

融合 ALS 个性化评分与动漫热门度：

HybridScore = alpha * ALSScore
            + (1 - alpha) * PopularityScore

融合前对相关分数进行归一化处理。

使用验证集比较五组权重：

0.00、0.25、0.50、0.75、1.00

以验证集 NDCG@10 选择最佳权重 alpha = 0.50。

### 4.3 最终排序评估

使用独立测试集，对 100 名活跃用户进行推荐评估。

| 算法 | Precision@10 | Recall@10 | HitRate@10 | NDCG@10 |
|---|---:|---:|---:|---:|
| ALS | 0.078 | 0.0406 | 0.52 | 0.0841 |
| Popularity | 0.189 | 0.0908 | 0.80 | 0.2115 |
| Hybrid | 0.193 | 0.0967 | 0.86 | 0.2217 |

在本次测试中，Hybrid 的 NDCG@10
相比热门推荐基线相对提升约 4.87%。

实验使用有限评分样本和活跃用户子集，
结果不能直接推广到全部用户。

## 5. 项目结构

- app.py：Streamlit Web 主程序
- src/01～03：数据分析与统计
- src/04：ALS 样本准备
- src/05：ALS 模型训练
- src/06～07：个性化推荐与批量导出
- src/08：ALS 参数优化
- src/09：推荐排序评估
- src/10：Hybrid 验证集实验
- src/11：Hybrid 独立测试集评估
- src/12：三策略推荐导出
- output/：实验结果及网页展示数据
- .streamlit/：网页主题配置
- requirements.txt：Python 依赖

## 6. 本地运行

环境要求：

- Python 3.12
- Java 17
- Python 依赖见 requirements.txt

安装依赖：

    python3 -m venv .venv
    source .venv/bin/activate
    pip install -r requirements.txt

如果需要复现实验，请先按照源代码准备清洗后的评分样本。

运行模型训练、验证和测试：

    python src/05_train_als.py
    python src/08_als_tuning.py
    python src/10_hybrid_validation.py
    python src/11_hybrid_final_test.py
    python src/12_export_recommendations.py

注意：部分程序依赖之前步骤生成的数据和模型。
请先准备 data/processed/ratings_als_sample.csv。

启动 Web：

    streamlit run app.py --server.port 8501

网页依赖 output/ 目录中的统计数据、评估结果及推荐 CSV。
仓库保留当前演示所需的小型输出文件。

## 7. 项目总结

本项目实现了从数据存储、统计分析、
机器学习模型训练到 Web 可视化的完整实验流程。

通过参数对比与推荐排序实验，发现显式评分 ALS
在当前数据下的 TOP 10 排序效果不及热门推荐。

进一步设计了结合个性化偏好与热门度的混合推荐策略，
并在独立测试集上观察到了排序效果提升。

目前仍存在样本规模有限、冷启动支持不足、
推荐结果以离线生成为主等局限。

## 8. 项目分工

独立完成：数据准备、Linux 环境搭建、数据处理、
算法实现、模型评估、Web 开发及项目展示。
