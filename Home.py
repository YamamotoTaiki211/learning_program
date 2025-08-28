import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(page_title="Order Analytics Dashboard", layout="wide")

# ===== Data Loading with Error Handling =====
@st.cache_data
def load_data():
    try:
        orders_df = pd.read_csv("sample_data/orders.csv")
        users_df = pd.read_csv("sample_data/users.csv")
        return orders_df, users_df, None
    except FileNotFoundError as e:
        return None, None, f"データファイルが見つかりません: {e}"
    except pd.errors.EmptyDataError:
        return None, None, "データファイルが空です"
    except Exception as e:
        return None, None, f"データ読み込みエラー: {e}"

# ===== Data Preprocessing =====
@st.cache_data
def preprocess_data(orders_df, users_df):
    try:
        # Orders data preprocessing
        orders_df['created_at'] = pd.to_datetime(orders_df['created_at'])
        orders_df['year_month'] = orders_df['created_at'].dt.to_period('M').astype(str)
        orders_df['is_cancelled'] = (orders_df['status'] == 'Cancelled').astype(int)
        orders_df['is_completed'] = (orders_df['status'] == 'Complete').astype(int)
        
        # Users data preprocessing  
        users_df['created_at'] = pd.to_datetime(users_df['created_at'])
        
        # Merge orders with users
        merged_df = orders_df.merge(users_df, left_on='user_id', right_on='id', suffixes=('_order', '_user'))
        
        return merged_df, None
    except Exception as e:
        return None, f"データ前処理エラー: {e}"

# ===== Monthly Analytics Calculation =====
@st.cache_data
def calculate_monthly_metrics(df):
    try:
        monthly_stats = df.groupby('year_month').agg({
            'order_id': 'count',
            'is_cancelled': 'sum',
            'is_completed': 'sum',
            'num_of_item': 'mean'
        }).reset_index()
        
        monthly_stats.columns = ['year_month', 'total_orders', 'cancelled_orders', 'completed_orders', 'avg_items']
        
        # Safe division for cancel rate
        monthly_stats['cancel_rate'] = np.where(
            monthly_stats['total_orders'] > 0,
            (monthly_stats['cancelled_orders'] / monthly_stats['total_orders'] * 100).round(2),
            0
        )
        
        # Calculate month-over-month growth
        monthly_stats['orders_growth'] = monthly_stats['total_orders'].pct_change() * 100
        
        return monthly_stats, None
    except Exception as e:
        return None, f"月次指標計算エラー: {e}"

# ===== Load and Process Data =====
orders_df, users_df, load_error = load_data()

if load_error:
    st.error(load_error)
    st.stop()

merged_df, preprocess_error = preprocess_data(orders_df, users_df)

if preprocess_error:
    st.error(preprocess_error)
    st.stop()

monthly_stats, calc_error = calculate_monthly_metrics(merged_df)

if calc_error:
    st.error(calc_error)
    st.stop()

# ===== Dashboard UI =====
st.title("📊 注文分析ダッシュボード")
st.markdown("**月別オーダー数推移とキャンセル率分析**")

# ===== Sidebar Filters =====
with st.sidebar:
    st.header("🎛️ フィルター")
    
    # Date range filter
    min_date = merged_df['created_at_order'].min().date()
    max_date = merged_df['created_at_order'].max().date()
    date_range = st.date_input(
        "期間選択",
        value=(min_date, max_date),
        min_value=min_date,
        max_value=max_date,
        format="YYYY-MM-DD"
    )
    
    # Region filter
    regions = sorted(merged_df['state'].dropna().unique())
    selected_regions = st.multiselect("地域", options=regions, default=regions[:5] if len(regions) > 5 else regions)
    
    # Traffic source filter
    sources = sorted(merged_df['traffic_source'].dropna().unique())
    selected_sources = st.multiselect("流入元", options=sources, default=sources)

# Filter data based on selections
start_date = pd.to_datetime(date_range[0])
end_date = pd.to_datetime(date_range[1])

filtered_df = merged_df[
    (merged_df['created_at_order'].between(start_date, end_date)) &
    (merged_df['state'].isin(selected_regions) if selected_regions else True) &
    (merged_df['traffic_source'].isin(selected_sources) if selected_sources else True)
]

if filtered_df.empty:
    st.warning("選択された条件に該当するデータがありません。フィルター条件を調整してください。")
    st.stop()

# Recalculate metrics for filtered data
filtered_monthly_stats, _ = calculate_monthly_metrics(filtered_df)

# ===== KPI Metrics =====
total_orders = int(filtered_df['order_id'].count())
total_cancelled = int(filtered_df['is_cancelled'].sum())
avg_cancel_rate = (total_cancelled / total_orders * 100) if total_orders > 0 else 0
avg_items = float(filtered_df['num_of_item'].mean())

col1, col2, col3, col4 = st.columns(4)
col1.metric("総注文数", f"{total_orders:,}")
col2.metric("キャンセル数", f"{total_cancelled:,}")
col3.metric("平均キャンセル率", f"{avg_cancel_rate:.1f}%")
col4.metric("平均商品数", f"{avg_items:.1f}個")

st.markdown("---")

# ===== Charts =====
if len(filtered_monthly_stats) > 0:
    # 1. 月別オーダー数の棒グラフ
    st.subheader("📊 月別オーダー数")
    fig_orders = px.bar(
        filtered_monthly_stats,
        x='year_month',
        y='total_orders',
        title='月別オーダー数推移',
        labels={'year_month': '年月', 'total_orders': 'オーダー数'}
    )
    fig_orders.update_xaxes(tickangle=45)
    st.plotly_chart(fig_orders, use_container_width=True)
    
    # 2. 月別キャンセル率の線グラフ
    st.subheader("📈 月別キャンセル率")
    fig_cancel = px.line(
        filtered_monthly_stats,
        x='year_month',
        y='cancel_rate',
        title='月別キャンセル率推移',
        labels={'year_month': '年月', 'cancel_rate': 'キャンセル率 (%)'},
        markers=True
    )
    fig_cancel.update_xaxes(tickangle=45)
    st.plotly_chart(fig_cancel, use_container_width=True)
else:
    st.warning("データがありません。")