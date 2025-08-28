# app.py
import io
import pandas as pd
import numpy as np
import plotly.express as px
import streamlit as st
from datetime import datetime

# ===== Page config =====
st.set_page_config(page_title="Sales BI", layout="wide")

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

st.title("売上BIダッシュボード")

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

# 3) 地域×チャネル ピボット（積み上げ棒）
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
        "- `revenue` は真値として扱い、再計算で上書きしない\n"
        "- 平均単価は集計後に `sum(revenue) / sum(units)` で算出\n"
        "- カテゴリ等は動的ユニーク値で選択肢を生成\n"
        "- 欠損は to_numeric(..., errors='coerce') で吸収\n"
        "- 乖離チェックは品質監視の参考（任意）"
    )
