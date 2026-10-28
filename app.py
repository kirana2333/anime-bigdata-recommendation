import html
import json
from pathlib import Path

import streamlit as st
import pandas as pd
import altair as alt

ROOT = Path(__file__).resolve().parent

st.set_page_config(
    page_title="Anime Data Lab",
    page_icon="🌌",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ========== 数据读取 ==========
@st.cache_data
def read_csv(path, modified_time):
    return pd.read_csv(path)

def load_file(path):
    if not path.exists():
        st.error(f"文件不存在：{path}")
        st.stop()
    return read_csv(str(path), path.stat().st_mtime_ns)

def load_stats(folder):
    files = sorted(
        (ROOT / "output" / folder).glob("part-*.csv")
    )
    if not files:
        st.error(f"找不到统计结果：{folder}")
        st.stop()
    return load_file(files[0])

# ========== 侧边栏 ==========
with st.sidebar:
    st.title("Anime Data Lab")
    st.caption("动漫数据分析与个性化推荐系统")

    st.divider()

    page = st.radio(
        "功能导航",
        [
            "数据总览",
            "动漫检索",
            "个性化推荐",
            "模型评估"
        ]
    )

    st.divider()
    st.caption("技术架构")
    st.write("Hadoop HDFS")
    st.write("Apache Spark SQL")
    st.write("Spark MLlib ALS")
    st.write("Python / Streamlit")

st.title("Anime Data Lab")
st.caption("基于 Hadoop 与 Spark 的动漫数据分析及个性化推荐系统")

# ========== 页面一：数据总览 ==========
if page == "数据总览":

    st.header("数据总览")

    type_df = load_stats("type_stats")
    genre_df = load_stats("genre_stats")

    c1, c2, c3 = st.columns(3)

    c1.metric(
        "有效评分动漫",
        f"{int(type_df['anime_count'].sum()):,}"
    )
    c2.metric("动画形式", len(type_df))
    c3.metric("展示的热门题材", len(genre_df))

    st.divider()

    st.subheader("动画形式分布")

    chart = alt.Chart(type_df).mark_bar(
        color="#A78BFA"
    ).encode(
        x=alt.X("type:N", title="动画形式"),
        y=alt.Y(
            "anime_count:Q",
            title="动漫数量",
            scale=alt.Scale(zero=True)
        ),
        tooltip=["type", "anime_count", "avg_rating"]
    )

    st.altair_chart(chart, width="stretch")
    st.dataframe(type_df, hide_index=True, width="stretch")

    st.divider()
    st.subheader("热门动漫题材 TOP 15")

    genre_chart = alt.Chart(genre_df).mark_bar(
        color="#818CF8"
    ).encode(
        x=alt.X(
            "anime_count:Q",
            title="动漫数量",
            scale=alt.Scale(zero=True)
        ),
        y=alt.Y(
            "single_genre:N",
            sort="-x",
            title="题材"
        ),
        tooltip=["single_genre", "anime_count"]
    )

    st.altair_chart(genre_chart, width="stretch")

    st.caption("一部动漫可能属于多个题材，各题材数量可以重叠。")


# ========== 页面二：动漫检索 ==========
elif page == "动漫检索":

    st.header("动漫搜索与筛选")

    anime = load_file(ROOT / "data/raw/anime.csv").copy()

    anime["name"] = anime["name"].fillna("").map(html.unescape)
    anime["genre"] = anime["genre"].fillna("")
    anime["type"] = anime["type"].fillna("Unknown")
    anime["rating"] = pd.to_numeric(
        anime["rating"], errors="coerce"
    )
    anime["members"] = pd.to_numeric(
        anime["members"], errors="coerce"
    )

    anime = anime[anime["rating"].between(0, 10)].copy()

    all_genres = sorted({
        genre.strip()
        for item in anime["genre"]
        for genre in item.split(",")
        if genre.strip()
    })

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        keyword = st.text_input(
            "动漫名称",
            placeholder="例如 Naruto"
        )
    with c2:
        selected_type = st.selectbox(
            "动画形式",
            ["全部"] + sorted(anime["type"].unique().tolist())
        )
    with c3:
        selected_genre = st.selectbox(
            "动漫题材",
            ["全部"] + all_genres
        )
    with c4:
        min_rating = st.slider(
            "最低评分",
            0.0, 10.0, 7.0, 0.5
        )

    filtered = anime.copy()

    if keyword:
        filtered = filtered[
            filtered["name"].str.contains(
                keyword, case=False, regex=False, na=False
            )
        ]

    if selected_type != "全部":
        filtered = filtered[
            filtered["type"] == selected_type
        ]

    if selected_genre != "全部":
        filtered = filtered[
            filtered["genre"].str.split(",").apply(
                lambda gs: selected_genre in
                [g.strip() for g in gs]
            )
        ]

    filtered = filtered[
        filtered["rating"] >= min_rating
    ]

    sort_by = st.selectbox(
        "排序方式",
        ["评分从高到低", "关注人数从高到低"]
    )

    if sort_by == "评分从高到低":
        filtered = filtered.sort_values(
            ["rating", "members"],
            ascending=[False, False]
        )
    else:
        filtered = filtered.sort_values(
            ["members", "rating"],
            ascending=[False, False]
        )

    st.metric("符合条件的动漫", len(filtered))

    st.dataframe(
        filtered[
            ["name", "type", "genre", "rating", "members"]
        ].head(50),
        hide_index=True,
        width="stretch"
    )

    st.caption("最多展示前 50 条筛选结果。")


# ========== 页面三：个性化推荐 ==========
elif page == "个性化推荐":

    st.header("混合个性化动漫推荐")

    rec = load_file(
        ROOT / "output/recommendations_strategy_multiuser.csv"
    )

    strategy_names = {
        "Hybrid": "Hybrid 混合推荐",
        "ALS": "ALS 协同过滤",
        "Popularity": "热门动漫推荐"
    }

    strategy = st.radio(
        "选择推荐算法",
        ["Hybrid", "ALS", "Popularity"],
        format_func=lambda x: strategy_names[x],
        horizontal=True
    )

    users = sorted(
        rec["user_id"].dropna().astype(int).unique().tolist()
    )

    selected_user = st.selectbox(
        "选择用户 ID",
        users,
        index=users.index(478) if 478 in users else 0
    )

    user_rec = rec[
        (rec["user_id"] == selected_user)
        & (rec["strategy"] == strategy)
    ].sort_values("rank").copy()

    st.subheader(
        f"用户 {selected_user} · {strategy_names[strategy]}"
    )

    c1, c2, c3 = st.columns(3)
    c1.metric("可选用户", len(users))
    c2.metric("推荐数量", len(user_rec))
    c3.metric("推荐策略", strategy)

    st.caption(
        "排序分数经过归一化处理，范围为0～1。"
        "分数用于各策略内部排序，"
        "不能直接解释为动漫的实际评分。"
    )

    user_rec["score"] = user_rec["score"].round(4)

    display = user_rec[
        ["rank", "name", "type", "genre", "score"]
    ].rename(columns={
        "rank": "排名",
        "name": "动漫名称",
        "type": "动画形式",
        "genre": "动漫题材",
        "score": "排序分数"
    })

    st.dataframe(
        display,
        hide_index=True,
        width="stretch"
    )

    st.info(
        "Hybrid 使用50% ALS 个性化分数与"
        "50% 热门度分数进行融合。"
        "可切换算法，观察同一用户的推荐差异。"
    )


# ========== 页面四：模型评估 ==========
elif page == "模型评估":

    st.header("Spark MLlib ALS 模型评估")

    tuning = load_file(
        ROOT / "output/als_tuning_results.csv"
    )

    summary_path = ROOT / "output/als_tuning_summary.json"

    if not summary_path.exists():
        st.error("未找到最终模型评估结果")
        st.stop()

    summary = json.loads(
        summary_path.read_text(encoding="utf-8")
    )

    c1, c2, c3 = st.columns(3)

    c1.metric(
        "最终测试 RMSE",
        f"{summary['test_rmse']:.4f}"
    )
    c2.metric(
        "最终测试 MAE",
        f"{summary['test_mae']:.4f}"
    )
    c3.metric(
        "测试集预测覆盖率",
        f"{summary['test_coverage']:.2%}"
    )

    improvement = (
        1 - summary["test_rmse"] / summary["baseline_rmse"]
    ) * 100

    st.success(
        f"相对于平均评分基线，"
        f"ALS 的测试 RMSE 降低了 {improvement:.2f}%"
    )

    st.divider()
    st.subheader("ALS 参数对比")

    tuning["参数组合"] = tuning.apply(
        lambda row:
        f"rank={int(row['rank'])}, λ={row['regParam']:.2f}",
        axis=1
    )

    tuning_chart = alt.Chart(tuning).mark_bar(
        color="#A78BFA"
    ).encode(
        x=alt.X(
            "参数组合:N",
            title="模型参数",
            sort=None,
            axis=alt.Axis(labelAngle=-20)
        ),
        y=alt.Y(
            "RMSE:Q",
            title="验证集 RMSE",
            scale=alt.Scale(zero=True)
        ),
        tooltip=[
            "参数组合",
            alt.Tooltip("RMSE:Q", format=".4f"),
            alt.Tooltip("MAE:Q", format=".4f"),
            alt.Tooltip("coverage:Q", format=".2%")
        ]
    )

    st.altair_chart(tuning_chart, width="stretch")

    st.dataframe(
        tuning[
            ["rank", "regParam", "RMSE", "MAE", "coverage"]
        ].round(4),
        hide_index=True,
        width="stretch"
    )

    st.divider()
    st.subheader("最优模型与基线对比")

    compare = pd.DataFrame({
        "模型": ["平均评分基线", "ALS 优化模型"],
        "RMSE": [
            summary["baseline_rmse"],
            summary["test_rmse"]
        ]
    })

    comparison_chart = alt.Chart(compare).mark_bar(
        color="#818CF8"
    ).encode(
        x=alt.X("模型:N", title="模型"),
        y=alt.Y(
            "RMSE:Q",
            title="测试集 RMSE",
            scale=alt.Scale(zero=True)
        ),
        tooltip=["模型", alt.Tooltip("RMSE:Q", format=".4f")]
    )

    st.altair_chart(comparison_chart, width="stretch")

    st.write(
        f"最优参数：rank={summary['best_rank']}，"
        f"regParam={summary['best_regParam']}"
    )

    st.caption(
        "参数由验证集选择，最终性能在独立测试集上计算。"
        "基线在相同的可预测测试记录上比较。"
    )

    # ===== 推荐排序效果评估 =====
    st.divider()
    st.subheader("推荐排序效果对比")

    ranking_path = ROOT / "output/ranking_evaluation.csv"

    if ranking_path.exists():
        ranking_df = load_file(ranking_path).copy()

        ranking_df["model"] = ranking_df["model"].replace({
            "ALS": "ALS 协同过滤",
            "Popularity": "热门推荐基线"
        })

        metric_map = {
            "Precision@10": "precision_at_10",
            "Recall@10": "recall_at_10",
            "HitRate@10": "hitrate_at_10",
            "NDCG@10": "ndcg_at_10"
        }

        selected_metric = st.selectbox(
            "选择推荐排序评价指标",
            list(metric_map.keys())
        )

        metric_col = metric_map[selected_metric]

        chart = alt.Chart(ranking_df).mark_bar().encode(
            x=alt.X(
                "model:N",
                title="推荐算法"
            ),
            y=alt.Y(
                f"{metric_col}:Q",
                title=selected_metric,
                scale=alt.Scale(zero=True)
            ),
            color=alt.Color(
                "model:N",
                scale=alt.Scale(
                    domain=[
                        "ALS 协同过滤",
                        "热门推荐基线"
                    ],
                    range=["#A78BFA", "#818CF8"]
                ),
                legend=None
            ),
            tooltip=[
                "model:N",
                alt.Tooltip(
                    f"{metric_col}:Q",
                    format=".4f",
                    title=selected_metric
                )
            ]
        )

        st.altair_chart(chart, width="stretch")

        display = ranking_df.rename(columns={
            "model": "模型",
            "precision_at_10": "Precision@10",
            "recall_at_10": "Recall@10",
            "hitrate_at_10": "HitRate@10",
            "ndcg_at_10": "NDCG@10"
        })

        st.dataframe(
            display.round(4),
            hide_index=True,
            width="stretch"
        )

        st.info(
            "在当前有限样本的排序实验中，"
            "热门推荐基线优于 ALS 协同过滤。"
            "这说明较低的评分预测误差"
            "并不必然带来更好的 TOP 10 推荐排序。"
        )

        st.caption(
            "排序实验采用独立构造的数据划分，"
            "使用100名活跃用户中的可评价用户进行测试。"
            "该实验的模型与网页部署的 ALS 模型分别训练，"
            "结果用于分析推荐策略的相对表现。"
        )

    else:
        st.warning("尚未找到推荐排序实验结果。")

    # ===== Hybrid 最终测试结果 =====
    st.divider()
    st.header("Hybrid 混合推荐最终测试")

    final_path = ROOT / "output/hybrid_final_results.csv"

    if final_path.exists():

        final_df = load_file(final_path).copy()

        final_df["model"] = final_df["model"].replace({
            "ALS": "ALS 协同过滤",
            "Popularity": "热门推荐",
            "Hybrid": "Hybrid 混合推荐"
        })

        st.subheader("独立测试集排序效果")

        metrics = {
            "Precision@10": "precision_at_10",
            "Recall@10": "recall_at_10",
            "HitRate@10": "hitrate_at_10",
            "NDCG@10": "ndcg_at_10"
        }

        selected = st.selectbox(
            "选择最终测试指标",
            list(metrics.keys()),
            key="hybrid_final_metric"
        )

        col = metrics[selected]

        chart = alt.Chart(final_df).mark_bar().encode(
            x=alt.X(
                f"{col}:Q",
                title=selected,
                scale=alt.Scale(zero=True)
            ),
            y=alt.Y(
                "model:N",
                title="推荐算法",
                sort="-x"
            ),
            color=alt.Color(
                "model:N",
                legend=None,
                scale=alt.Scale(
                    domain=[
                        "ALS 协同过滤",
                        "热门推荐",
                        "Hybrid 混合推荐"
                    ],
                    range=[
                        "#818CF8",
                        "#64748B",
                        "#A78BFA"
                    ]
                )
            ),
            tooltip=[
                "model:N",
                alt.Tooltip(
                    f"{col}:Q",
                    format=".4f",
                    title=selected
                )
            ]
        )

        st.altair_chart(chart, width="stretch")

        display = final_df.rename(columns={
            "model": "算法",
            "precision_at_10": "Precision@10",
            "recall_at_10": "Recall@10",
            "hitrate_at_10": "HitRate@10",
            "ndcg_at_10": "NDCG@10"
        })

        st.dataframe(
            display.round(4),
            hide_index=True,
            width="stretch"
        )

        popular = final_df[
            final_df["model"] == "热门推荐"
        ]["ndcg_at_10"].iloc[0]

        hybrid = final_df[
            final_df["model"] == "Hybrid 混合推荐"
        ]["ndcg_at_10"].iloc[0]

        improvement = (hybrid / popular - 1) * 100

        st.metric(
            "Hybrid 相比热门推荐的 NDCG@10 相对变化",
            f"{improvement:+.2f}%"
        )

        st.caption(
            "混合权重通过验证集选择，"
            "最终结果来自独立测试集。"
            "实验针对100名活跃用户，"
            "结论限于当前样本和评估条件。"
        )

    else:
        st.warning("尚未找到 Hybrid 最终测试数据。")
