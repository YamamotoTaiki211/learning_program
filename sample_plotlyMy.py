import streamlit as st
import pandas as pd
import plotly.graph_objects as go

# -----------------------------------
# タイトルと説明
# -----------------------------------
st.title('Plotly基礎')
st.write('Plotlyを使ってインタラクティブなグラフを作成してみましょう！')

# -----------------------------------
# CSVファイルの読み込み
# -----------------------------------
df = pd.read_csv('data/sample_sales.csv')

# -----------------------------------
# カテゴリ別売上の合計を計算
# -----------------------------------
category_revenue = df.groupby('category')['revenue'].sum().reset_index()
categories = category_revenue['category'].tolist()
revenues = category_revenue['revenue'].tolist()

# -----------------------------------
# アニメーション用のフレームを作成（1本ずつ増やしていく）
frames = []
for i in range(1, len(categories) + 1):
    frames.append(go.Frame(
        data=[
            go.Bar(
                x=categories,
                y=[revenues[j] if j < i else 0 for j in range(len(categories))],
                marker_color='green'
            )
        ],
        name=f'frame{i}'
    ))

# -----------------------------------
# 初期状態（すべて高さ0のバー）
initial_y = [0] * len(categories)
fig = go.Figure(
    data=[
        go.Bar(
            x=categories,
            y=initial_y,
            marker_color='green'
        )
    ],
    layout=go.Layout(
        title='商品カテゴリごとの総売上',
        xaxis_title='商品カテゴリ',
        yaxis_title='総売上 (円)',
        yaxis=dict(range=[0, max(revenues) * 1.1]),
        updatemenus=[{
            'type': 'buttons',
            'buttons': [{
                'label': 'Play',
                'method': 'animate',
                'args': [None, {
                    'frame': {'duration': 500, 'redraw': True},
                    'fromcurrent': True,
                    'transition': {'duration': 300, 'easing': 'linear'}
                }]
            }]
        }]
    ),
    frames=frames
)

# -----------------------------------
# グラフの表示（初期状態）
st.plotly_chart(fig, use_container_width=True)

# 説明
st.write('---')
st.write('▶ボタンを押すと、バーが1本ずつ順ににょきにょき伸びていきます。')
