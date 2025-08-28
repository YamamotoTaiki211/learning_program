import streamlit as st
import pandas as pd
import plotly.express as px

# アプリのタイトルと説明
st.title('Plotly基礎')
st.write('Plotlyを使ってインタラクティブなグラフを作成してみましょう！')

# CSV読込（列名の違いに強くするため、まずは普通に読み込む）
df = pd.read_csv('data/sample_sales.csv')

# 列名の候補（あなたのCSVに合わせて自動で選択）
def pick(cols, candidates):
    return next((c for c in candidates if c in cols), None)

cat_col = pick(df.columns, ['category', 'category_name'])
rev_col = pick(df.columns, ['revenue', 'revenue_total'])
date_col = pick(df.columns, ['date', 'created_at'])

# 日付列があればdatetimeに変換（なくてもOK）
if date_col:
    try:
        df[date_col] = pd.to_datetime(df[date_col])
    except Exception as e:
        st.warning(f'日付列 {date_col} の変換に失敗しました: {e}')

# 必要列があるかチェック
missing = []
if not cat_col:
    missing.append("category / category_name")
if not rev_col:
    missing.append("revenue / revenue_total")

if missing:
    st.error("必要な列が見つかりません: " + ", ".join(missing))
    st.stop()

st.subheader('カテゴリ別合計売上グラフ')

# カテゴリ別の合計売上
category_revenue = df.groupby(cat_col)[rev_col].sum().reset_index()

# 棒グラフ
fig = px.bar(
    category_revenue,
    x=cat_col,
    y=rev_col,
    title='商品カテゴリごとの総売上',
    labels={cat_col: '商品カテゴリ', rev_col: '総売上 (円)'}
)

st.plotly_chart(fig, use_container_width=True)

st.write('---')
st.write('このグラフはインタラクティブです！特定のカテゴリにカーソルを合わせると、そのカテゴリの正確な総売上が表示されます。')
