# dashboard_enhanced.py - ヒートマップと月次トレンド分析を追加したバージョン
import io
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from datetime import datetime

# ===== Page config =====
st.set_page_config(page_title="Sales BI Enhanced", layout="wide")

# ===== Constants =====
DATA_PATH = "data/sample_sales.csv"
COLS = {
    "date": "date",
    "category": "category",
    "units": "units",
    "unit_price": "unit_price",
    "region": "region",
    "sales_channel": "sales_channel",
    "customer_segment": "customer_segment",
    "revenue": "revenue",
}

# ===== Data Load (cached) =====
@st.cache_data
def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(
        path,
        dtype={
            COLS["category"]: "string",
            COLS["region"]: "string",
            COLS["sales_channel"]: "string",
            COLS["customer_segment"]: "string",
        },
        parse_dates=[COLS["date"]],
        encoding="utf-8",
    )
    # Cast to category for performance
    for c in [COLS["category"], COLS["region"], COLS["sales_channel"], COLS["customer_segment"]]:
        df[c] = df[c].astype("category")

    # Numeric coercion
    for c in [COLS["units"], COLS["unit_price"], COLS["revenue"]]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # Simple validation
    missing = [c for c in COLS.values() if c not in df.columns]
    if missing:
        raise ValueError(f"想定カラムが見つかりません: {missing}")

    return df

try:
    df = load_data(DATA_PATH)
except Exception as e:
    st.error(f"データ読み込みに失敗しました: {e}")
    st.stop()

st.title("売上BIダッシュボード（強化版）")

# ===== Sidebar Filters =====
with st.sidebar:
    st.header("フィルター")

    # Date range from data min/max
    min_date = pd.to_datetime(df[COLS["date"]].min())
    max_date = pd.to_datetime(df[COLS["date"]].max())
    date_range = st.date_input(
        "期間",
        value=(min_date.date(), max_date.date()),
        min_value=min_date.date(),
        max_value=max_date.date(),
        format="YYYY-MM-DD",
    )

    cats = st.multiselect("カテゴリ", options=sorted(df[COLS["category"]].cat.categories.tolist()))
    regs = st.multiselect("地域", options=sorted(df[COLS["region"]].cat.categories.tolist()))
    chans = st.multiselect("販売チャネル", options=sorted(df[COLS["sales_channel"]].cat.categories.tolist()))
    segs = st.multiselect("顧客セグメント", options=sorted(df[COLS["customer_segment"]].cat.categories.tolist()))

    reset = st.button("フィルタをリセット")

if reset:
    # Rerun to clear widget states
    st.session_state.clear()
    st.rerun()

# ===== Filtering (pure) =====
start_dt = pd.to_datetime(date_range[0])
end_dt = pd.to_datetime(date_range[1])
mask = (
    df[COLS["date"]].between(start_dt, end_dt)
    & (df[COLS["category"]].isin(cats) if cats else True)
    & (df[COLS["region"]].isin(regs) if regs else True)
    & (df[COLS["sales_channel"]].isin(chans) if chans else True)
    & (df[COLS["customer_segment"]].isin(segs) if segs else True)
)
fdf = df.loc[mask].copy()

if fdf.empty:
    st.info("該当データがありません。フィルタ条件を緩めてください。")
    st.stop()

# ===== KPI =====
total_revenue = float(fdf[COLS["revenue"]].sum())
total_units = float(fdf[COLS["units"]].sum())
avg_price = (total_revenue / total_units) if total_units else np.nan
records = int(len(fdf))

# Optional quality check
fdf["calc_revenue"] = fdf[COLS["units"]] * fdf[COLS["unit_price"]]
diff = fdf[COLS["revenue"]] - fdf["calc_revenue"]
mismatch_cnt = int((diff.fillna(0) != 0).sum())
max_dev = float((diff.abs() / fdf[COLS["revenue"]].replace(0, np.nan)).max())

col1, col2, col3, col4, col5, col6 = st.columns(6)
col1.metric("合計売上(円)", f"{total_revenue:,.0f}")
col2.metric("合計数量", f"{total_units:,.0f}")
col3.metric("平均単価(円)", "-" if np.isnan(avg_price) else f"{avg_price:,.0f}")
col4.metric("レコード数", f"{records:,}")
col5.metric("乖離件数", f"{mismatch_cnt:,}")
col6.metric("最大乖離率", "-" if np.isinf(max_dev) or np.isnan(max_dev) else f"{max_dev*100:,.2f}%")

st.markdown("---")

# ===== Charts =====
# 1) 日次売上推移
ts = (
    fdf.groupby(COLS["date"], as_index=False)[COLS["revenue"]]
    .sum()
    .sort_values(COLS["date"])
)
fig_ts = px.line(
    ts, x=COLS["date"], y=COLS["revenue"],
    title="日次売上推移", labels={COLS["date"]: "日付", COLS["revenue"]: "売上(円)"}
)
st.plotly_chart(fig_ts, use_container_width=True)

# 分析解釈
if len(ts) > 1:
    max_day = ts.loc[ts[COLS["revenue"]].idxmax()]
    min_day = ts.loc[ts[COLS["revenue"]].idxmin()]
    avg_daily = ts[COLS["revenue"]].mean()
    trend = "上昇" if ts[COLS["revenue"]].iloc[-1] > ts[COLS["revenue"]].iloc[0] else "下降"
    
    st.info(f"""
    📊 **日次売上推移の分析:**
    - 最高売上日: {max_day[COLS['date']].strftime('%Y-%m-%d')} ({max_day[COLS['revenue']]:,.0f}円)
    - 最低売上日: {min_day[COLS['date']].strftime('%Y-%m-%d')} ({min_day[COLS['revenue']]:,.0f}円)
    - 平均日次売上: {avg_daily:,.0f}円
    - 期間全体のトレンド: {trend}傾向
    """)
else:
    st.info("📊 **日次売上推移:** データが1日分のみのため、トレンド分析はできません。")

# 2) カテゴリ別売上
cat_rev = (
    fdf.groupby(COLS["category"], as_index=False)[COLS["revenue"]]
    .sum()
    .sort_values(COLS["revenue"], ascending=False)
)
fig_cat = px.bar(
    cat_rev, x=COLS["category"], y=COLS["revenue"],
    title="カテゴリ別売上", labels={COLS["category"]: "カテゴリ", COLS["revenue"]: "売上(円)"}
)
st.plotly_chart(fig_cat, use_container_width=True)

# 分析解釈
if len(cat_rev) > 0:
    top_cat = cat_rev.iloc[0]
    total_cat_revenue = cat_rev[COLS["revenue"]].sum()
    top_cat_share = (top_cat[COLS["revenue"]] / total_cat_revenue * 100)
    
    analysis_text = f"""
    📊 **カテゴリ別売上の分析:**
    - 最高売上カテゴリ: **{top_cat[COLS['category']]}** ({top_cat[COLS['revenue']]:,.0f}円, {top_cat_share:.1f}%)
    - カテゴリ数: {len(cat_rev)}種類
    """
    
    if len(cat_rev) > 1:
        second_cat = cat_rev.iloc[1]
        analysis_text += f"- 2位: **{second_cat[COLS['category']]}** ({second_cat[COLS['revenue']]:,.0f}円)"
    
    st.info(analysis_text)

# 3) 新機能: 月次売上トレンド分析
fdf_monthly = fdf.copy()
fdf_monthly['年月'] = fdf_monthly[COLS["date"]].dt.to_period('M').astype(str)
monthly_trend = (
    fdf_monthly.groupby(['年月', COLS["region"]], as_index=False)[COLS["revenue"]]
    .sum()
    .sort_values('年月')
)
fig_monthly = px.line(
    monthly_trend, x='年月', y=COLS["revenue"], color=COLS["region"],
    title="月次売上トレンド（地域別）", 
    labels={'年月': '年月', COLS["revenue"]: '売上(円)', COLS["region"]: '地域'}
)
fig_monthly.update_xaxes(tickangle=45)
st.plotly_chart(fig_monthly, use_container_width=True)

# 分析解釈
if len(monthly_trend) > 0:
    regions_in_data = monthly_trend[COLS["region"]].unique()
    best_region_month = monthly_trend.loc[monthly_trend[COLS["revenue"]].idxmax()]
    
    # 地域別の月次パフォーマンス
    region_performance = monthly_trend.groupby(COLS["region"])[COLS["revenue"]].agg(['sum', 'mean', 'std']).round(0)
    best_region = region_performance['sum'].idxmax()
    
    analysis_text = f"""
    📊 **月次売上トレンドの分析:**
    - 最高売上月: **{best_region_month['年月']}** の {best_region_month[COLS['region']]} ({best_region_month[COLS['revenue']]:,.0f}円)
    - 最高売上地域: **{best_region}** (累計: {region_performance.loc[best_region, 'sum']:,.0f}円)
    - 対象地域数: {len(regions_in_data)}地域
    """
    
    if len(region_performance) > 1:
        most_stable = region_performance['std'].idxmin()
        analysis_text += f"- 最も安定した地域: **{most_stable}** (標準偏差: {region_performance.loc[most_stable, 'std']:,.0f})"
    
    st.info(analysis_text)

# 4) 新機能: ヒートマップ（地域×曜日の売上密度）
fdf_heatmap = fdf.copy()
fdf_heatmap['曜日'] = fdf_heatmap[COLS["date"]].dt.day_name()
heatmap_data = (
    fdf_heatmap.groupby([COLS["region"], '曜日'], as_index=False)[COLS["revenue"]]
    .sum()
    .pivot(index=COLS["region"], columns='曜日', values=COLS["revenue"])
    .fillna(0)
)

# 曜日の順序を調整
weekday_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
heatmap_data = heatmap_data.reindex(columns=[day for day in weekday_order if day in heatmap_data.columns])

fig_heatmap = go.Figure(data=go.Heatmap(
    z=heatmap_data.values,
    x=heatmap_data.columns,
    y=heatmap_data.index,
    colorscale='Blues',
    text=heatmap_data.values,
    texttemplate='%{text:,.0f}',
    textfont={"size": 10},
    hoverongaps=False
))
fig_heatmap.update_layout(
    title="売上ヒートマップ（地域×曜日）",
    xaxis_title="曜日",
    yaxis_title="地域"
)
st.plotly_chart(fig_heatmap, use_container_width=True)

# 分析解釈
if not heatmap_data.empty:
    # 最高売上の組み合わせを見つける
    max_value = heatmap_data.values.max()
    max_pos = np.unravel_index(heatmap_data.values.argmax(), heatmap_data.shape)
    best_region = heatmap_data.index[max_pos[0]]
    best_day = heatmap_data.columns[max_pos[1]]
    
    # 各地域の最も売上が高い曜日
    best_days_by_region = heatmap_data.idxmax(axis=1)
    
    # 各曜日の売上合計
    daily_totals = heatmap_data.sum(axis=0).sort_values(ascending=False)
    best_overall_day = daily_totals.index[0]
    
    analysis_text = f"""
    📊 **売上ヒートマップの分析:**
    - 最高売上の組み合わせ: **{best_region}** の **{best_day}** ({max_value:,.0f}円)
    - 全体で最も売上が高い曜日: **{best_overall_day}** ({daily_totals.iloc[0]:,.0f}円)
    - 分析対象: {len(heatmap_data)}地域 × {len(heatmap_data.columns)}曜日
    """
    
    # 地域ごとの最適曜日を表示
    if len(best_days_by_region) > 1:
        region_insights = []
        for region, day in best_days_by_region.items():
            revenue = heatmap_data.loc[region, day]
            if revenue > 0:
                region_insights.append(f"{region}: {day}")
        
        if region_insights:
            analysis_text += f"\n    - 地域別最適曜日: {', '.join(region_insights[:3])}"
            if len(region_insights) > 3:
                analysis_text += f" など"
    
    st.info(analysis_text)

# 5) 地域×チャネル ピボット（積み上げ棒）
pivot = (
    fdf.pivot_table(
        index=COLS["region"],
        columns=COLS["sales_channel"],
        values=COLS["revenue"],
        aggfunc="sum",
        fill_value=0,
    )
    .reset_index()
)
pivot_melt = pivot.melt(id_vars=COLS["region"], var_name="販売チャネル", value_name="売上(円)")
fig_pv = px.bar(
    pivot_melt, x=COLS["region"], y="売上(円)", color="販売チャネル",
    title="地域×販売チャネルの売上（積み上げ）", labels={COLS["region"]: "地域"}
)
st.plotly_chart(fig_pv, use_container_width=True)

# 分析解釈
if len(pivot_melt) > 0:
    # 地域別総売上
    region_totals = pivot_melt.groupby(COLS["region"])["売上(円)"].sum().sort_values(ascending=False)
    best_region_channel = region_totals.index[0]
    
    # チャネル別総売上
    channel_totals = pivot_melt.groupby("販売チャネル")["売上(円)"].sum().sort_values(ascending=False)
    best_channel = channel_totals.index[0]
    
    # 最高売上の組み合わせ
    best_combination = pivot_melt.loc[pivot_melt["売上(円)"].idxmax()]
    
    analysis_text = f"""
    📊 **地域×販売チャネルの分析:**
    - 最高売上地域: **{best_region_channel}** ({region_totals.iloc[0]:,.0f}円)
    - 最高売上チャネル: **{best_channel}** ({channel_totals.iloc[0]:,.0f}円)
    - 最強の組み合わせ: **{best_combination[COLS['region']]}** × **{best_combination['販売チャネル']}** ({best_combination['売上(円)']:,.0f}円)
    - 分析対象: {len(region_totals)}地域 × {len(channel_totals)}チャネル
    """
    
    st.info(analysis_text)

st.markdown("---")

# ===== Data Table & Download =====
st.subheader("明細（現在のフィルタ反映）")
st.dataframe(fdf.drop(columns=["calc_revenue"]), use_container_width=True)

csv_bytes = fdf.drop(columns=["calc_revenue"]).to_csv(index=False).encode("utf-8-sig")
st.download_button(
    "CSVをダウンロード",
    data=csv_bytes,
    file_name=f"sales_filtered_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
    mime="text/csv",
)

# ===== Notes =====
with st.expander("実装メモ"):
    st.write(
        "**Enhanced Version 追加機能:**\n"
        "- 月次売上トレンド（地域別ライン）\n"
        "- ヒートマップ（地域×曜日の売上密度）\n\n"
        "**既存機能:**\n"
        "- `revenue` は真値として扱い、再計算で上書きしない\n"
        "- 平均単価は集計後に `sum(revenue) / sum(units)` で算出\n"
        "- カテゴリ等は動的ユニーク値で選択肢を生成\n"
        "- 欠損は to_numeric(..., errors='coerce') で吸収\n"
        "- 乖離チェックは品質監視の参考（任意）"
    )