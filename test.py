# 📚 必要なライブラリをインポート
import streamlit as st      # Webアプリケーション作成のためのメインライブラリ
import pandas as pd         # データ操作・分析ライブラリ
import numpy as np          # 数値計算ライブラリ
import plotly.express as px # グラフ作成ライブラリ（簡単な記法）
import plotly.graph_objects as go # グラフ作成ライブラリ（詳細設定可能）
from datetime import datetime, timedelta # 日付・時間処理
import random              # ランダム数値生成

# 🎨 ページの基本設定
st.set_page_config(
    page_title="実験用ダッシュボード",  # ブラウザタブに表示されるタイトル
    page_icon="🧪",                    # ブラウザタブに表示されるアイコン
    layout="wide"                      # ページレイアウトを幅広に設定
)

# 📝 メインタイトルと説明
st.title("🧪 Streamlit実験用ダッシュボード")
st.markdown("""
**このアプリについて：**
- Streamlitの様々な機能を試すことができる実験用アプリです
- 4つのタブで異なる機能を体験できます
- 各機能には詳しい説明を記載しているので、初心者の方も安心してお使いください

**使い方：** 下のタブをクリックして、各機能を試してみてください！
""")
st.markdown("---")

# 📋 タブの作成（複数のページを1つのアプリに統合）
tab1, tab2, tab3, tab4 = st.tabs(["📊 データ可視化", "🎛️ インタラクティブ要素", "📈 チャート実験", "🔢 データ処理"])

# 📊 タブ1: データ可視化の基本
with tab1:
    st.header("📊 データ可視化実験")
    st.markdown("""
    **ここで学べること：**
    - データフレーム（表形式データ）の表示方法
    - 基本的な統計値の表示方法  
    - 簡単な折れ線グラフの作成方法
    """)
    
    # 💡 2列レイアウトの作成
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📋 サンプルデータ")
        st.markdown("**説明：** ここではPandasを使って架空の売上データを作成し、表として表示します")
        
        # 🎲 ランダムなサンプルデータの作成
        sample_data = pd.DataFrame({
            '日付': pd.date_range('2024-01-01', periods=30, freq='D'),  # 30日分の日付
            '売上': np.random.normal(1000, 200, 30),                    # 平均1000、標準偏差200の正規分布
            '訪問者数': np.random.poisson(50, 30),                      # 平均50のポアソン分布
            'カテゴリ': np.random.choice(['A', 'B', 'C'], 30)           # A,B,Cからランダム選択
        })
        
        # 📊 データフレームの表示
        st.dataframe(sample_data, use_container_width=True)
        st.caption("💡 ヒント: 表は横スクロール可能で、列をクリックするとソートできます")
    
    with col2:
        st.subheader("📈 基本統計")
        st.markdown("**説明：** データから計算した統計値をカード形式で表示します")
        
        # 📊 メトリクス（統計値）の表示
        st.metric(
            label="平均売上", 
            value=f"¥{sample_data['売上'].mean():.0f}",  # 平均値
            delta=f"{sample_data['売上'].std():.0f}"      # 標準偏差（変動の目安）
        )
        st.metric(
            label="総訪問者数", 
            value=f"{sample_data['訪問者数'].sum()}",     # 合計値
            delta=f"{sample_data['訪問者数'].mean():.1f}/日"  # 1日平均
        )
        
        st.markdown("**📝 統計値の意味：**")
        st.markdown("- **平均売上**: 30日間の売上の平均値")
        st.markdown("- **標準偏差**: 売上のばらつきの大きさ")
        st.markdown("- **総訪問者数**: 30日間の訪問者数の合計")
        
    # 📈 折れ線グラフの作成と表示
    st.subheader("📊 売上推移グラフ")
    st.markdown("**説明：** Plotlyを使って売上の時系列変化を可視化します")
    
    fig = px.line(sample_data, x='日付', y='売上', title='📈 30日間の売上推移')
    fig.update_layout(
        xaxis_title="日付",
        yaxis_title="売上（円）",
        hovermode='x unified'  # ホバー情報を整理して表示
    )
    st.plotly_chart(fig, use_container_width=True)
    st.caption("💡 ヒント: グラフにマウスを合わせると詳細データが表示されます")

# 🎛️ タブ2: インタラクティブ要素の基本
with tab2:
    st.header("🎛️ インタラクティブ要素実験")
    st.markdown("""
    **ここで学べること：**
    - ユーザー入力を受け取る様々なウィジェット（部品）の使い方
    - ウィジェットの値を取得して処理に活用する方法
    - リアルタイムでページが更新される仕組み
    """)
    
    # 🏗️ 3列レイアウトの作成
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.subheader("🎚️ スライダー＆セレクトボックス")
        
        # スライダー（数値選択）
        st.markdown("**スライダー** - マウスで直感的に数値を選択")
        slider_val = st.slider("数値選択", 0, 100, 50)  # 最小値0、最大値100、初期値50
        st.write(f"選択値: {slider_val}")
        st.caption("💡 ドラッグして値を変更してみてください")
        
        # セレクトボックス（ドロップダウンリスト）
        st.markdown("**セレクトボックス** - リストから1つを選択")
        select_val = st.selectbox("オプション選択", ['オプション1', 'オプション2', 'オプション3'])
        st.write(f"選択: {select_val}")
        st.caption("💡 クリックして選択肢を表示")
    
    with col2:
        st.subheader("☑️ チェック＆ラジオボタン")
        
        # チェックボックス（ON/OFF切り替え）
        st.markdown("**チェックボックス** - ON/OFFの切り替え")
        checkbox_val = st.checkbox("チェックボックス")
        if checkbox_val:
            st.success("✅ チェックされています！")
        else:
            st.info("⬜ チェックされていません")
        st.caption("💡 クリックでON/OFFを切り替え")
        
        # ラジオボタン（複数選択肢から1つ）
        st.markdown("**ラジオボタン** - 複数選択肢から1つを選択")
        radio_val = st.radio("ラジオボタン", ['選択肢A', '選択肢B', '選択肢C'])
        st.write(f"選択: {radio_val}")
        st.caption("💡 1つだけ選択可能")
    
    with col3:
        st.subheader("📝 テキスト＆数値入力")
        
        # テキスト入力
        st.markdown("**テキスト入力** - 自由な文字列を入力")
        text_input = st.text_input("テキスト入力", placeholder="何か入力してください")
        if text_input:
            st.info(f"入力値: {text_input}")
            st.write(f"文字数: {len(text_input)}文字")
        st.caption("💡 何でも入力してみてください")
        
        # 数値入力
        st.markdown("**数値入力** - 数字のみを入力")
        number_input = st.number_input("数値入力", min_value=0, max_value=1000, value=100)
        st.write(f"数値: {number_input}")
        st.write(f"2倍: {number_input * 2}")
        st.caption("💡 上下ボタンでも調整可能")
    
    # 🎉 アクションボタン
    st.markdown("---")
    st.subheader("🔘 アクションボタン")
    st.markdown("**説明：** ボタンをクリックすると特定の処理を実行できます")
    
    col_btn1, col_btn2, col_btn3 = st.columns(3)
    
    with col_btn1:
        if st.button("🎈 バルーン"):
            st.balloons()  # 画面に風船アニメーション
            st.success("バルーンが飛びました！")
    
    with col_btn2:
        if st.button("🎊 雪"):
            st.snow()  # 画面に雪アニメーション
            st.success("雪が降りました！")
    
    with col_btn3:
        if st.button("🔢 計算実行"):
            result = slider_val * number_input
            st.info(f"スライダー値 × 数値入力 = {result}")
    
    st.caption("💡 ヒント: ボタンをクリックするとページが再読み込みされ、全ての値が更新されます")

# 📈 タブ3: 各種チャートの作成
with tab3:
    st.header("📈 チャート実験")
    st.markdown("""
    **ここで学べること：**
    - Plotlyを使った様々なグラフの作成方法
    - 線グラフ、棒グラフ、散布図、円グラフの特徴と使い分け
    - インタラクティブなグラフの作り方
    """)
    
    # 🎛️ グラフタイプ選択
    st.subheader("🎨 グラフタイプを選択してください")
    chart_type = st.selectbox(
        "チャートタイプ", 
        ['線グラフ', '棒グラフ', '散布図', '円グラフ'],
        help="選択したグラフタイプに応じて、異なるサンプルデータを表示します"
    )
    
    # 🎲 ランダムデータの生成
    x_data = np.random.randn(100)  # 標準正規分布（平均0、標準偏差1）
    y_data = np.random.randn(100)
    
    # 📊 選択されたグラフタイプに応じてグラフを表示
    if chart_type == '線グラフ':
        st.subheader("📈 線グラフ（時系列データ）")
        st.markdown("""
        **線グラフの特徴：**
        - 時間の変化や連続的なデータの推移を表示
        - トレンド（傾向）を把握しやすい
        - 複数の系列を重ねて比較可能
        """)
        
        fig = go.Figure()
        # ランダムウォーク（累積和）を作成
        cumulative_data = np.cumsum(x_data)
        fig.add_trace(go.Scatter(
            x=list(range(len(x_data))), 
            y=cumulative_data, 
            mode='lines+markers',
            name='ランダムウォーク',
            line=dict(color='blue', width=2)
        ))
        fig.update_layout(
            title='📈 ランダムウォーク（株価のような動き）',
            xaxis_title='時間',
            yaxis_title='値',
            hovermode='x unified'
        )
        st.plotly_chart(fig, use_container_width=True)
        st.caption("💡 このグラフは株価や気温変化のような連続データを模擬しています")
    
    elif chart_type == '棒グラフ':
        st.subheader("📊 棒グラフ（カテゴリ別データ）")
        st.markdown("""
        **棒グラフの特徴：**
        - カテゴリ別の数値を比較
        - 大小関係が一目で分かる
        - 売上、人口、評価など離散的なデータに最適
        """)
        
        categories = ['製品A', '製品B', '製品C', '製品D', '製品E']
        values = np.random.randint(10, 100, 5)
        
        fig = px.bar(
            x=categories, 
            y=values, 
            title='📊 製品別売上高',
            color=values,
            color_continuous_scale='viridis'
        )
        fig.update_layout(
            xaxis_title='製品',
            yaxis_title='売上高（万円）'
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # 最大・最小値の表示
        max_idx = np.argmax(values)
        min_idx = np.argmin(values)
        st.info(f"📈 最高売上: {categories[max_idx]} - {values[max_idx]}万円")
        st.info(f"📉 最低売上: {categories[min_idx]} - {values[min_idx]}万円")
    
    elif chart_type == '散布図':
        st.subheader("🔍 散布図（相関関係の可視化）")
        st.markdown("""
        **散布図の特徴：**
        - 2つの変数の関係性を可視化
        - 相関関係（正の相関、負の相関、無相関）を確認
        - 外れ値（異常値）の発見に有効
        """)
        
        # より関係性が分かりやすいデータを生成
        x_scatter = np.random.randn(100)
        y_scatter = x_scatter * 0.7 + np.random.randn(100) * 0.5  # 正の相関を持つデータ
        
        fig = px.scatter(
            x=x_scatter, 
            y=y_scatter, 
            title='🔍 2つの変数の関係性',
            opacity=0.7
        )
        fig.update_layout(
            xaxis_title='変数X',
            yaxis_title='変数Y'
        )
        
        # 近似直線を追加
        z = np.polyfit(x_scatter, y_scatter, 1)
        line_x = np.linspace(x_scatter.min(), x_scatter.max(), 100)
        line_y = z[0] * line_x + z[1]
        fig.add_trace(go.Scatter(
            x=line_x, 
            y=line_y, 
            mode='lines', 
            name='近似直線',
            line=dict(color='red', dash='dash')
        ))
        
        st.plotly_chart(fig, use_container_width=True)
        
        # 相関係数を計算・表示
        correlation = np.corrcoef(x_scatter, y_scatter)[0, 1]
        st.metric("相関係数", f"{correlation:.3f}")
        if correlation > 0.7:
            st.success("🔺 強い正の相関があります")
        elif correlation < -0.7:
            st.success("🔻 強い負の相関があります")
        else:
            st.info("➖ 弱い相関または無相関です")
    
    elif chart_type == '円グラフ':
        st.subheader("🥧 円グラフ（全体に占める割合）")
        st.markdown("""
        **円グラフの特徴：**
        - 全体に占める各部分の割合を表示
        - 構成比や市場シェアの可視化に最適
        - 項目数は5個以下が見やすい
        """)
        
        labels = ['モバイル', 'デスクトップ', 'タブレット', 'その他']
        values = np.random.randint(10, 50, 4)
        
        fig = px.pie(
            values=values, 
            names=labels, 
            title='🌐 デバイス別アクセス比率',
            hole=0.3  # ドーナツ型にする
        )
        fig.update_traces(textposition='inside', textinfo='percent+label')
        st.plotly_chart(fig, use_container_width=True)
        
        # 詳細情報を表示
        total = sum(values)
        st.markdown("**📊 詳細情報:**")
        for label, value in zip(labels, values):
            percentage = (value / total) * 100
            st.write(f"- **{label}**: {value} ({percentage:.1f}%)")
        
        st.caption("💡 円グラフは全体を100%として、各カテゴリの構成比を視覚的に表現します")

# 🔢 タブ4: データ処理とファイル操作
with tab4:
    st.header("🔢 データ処理実験")
    st.markdown("""
    **ここで学べること：**
    - CSVファイルのアップロード・読み込み方法
    - データフレームの基本操作と情報表示
    - ランダムデータの生成とダウンロード機能
    """)
    
    # 📁 ファイルアップロード機能
    st.subheader("📁 CSVファイルアップロード")
    st.markdown("""
    **CSVファイルとは？**
    - Comma Separated Values（カンマ区切り値）の略
    - ExcelやGoogleスプレッドシートで作成・保存可能
    - データ分析でよく使われる形式
    """)
    
    uploaded_file = st.file_uploader(
        "CSVファイルを選択してください", 
        type=['csv'],
        help="ExcelからCSV形式で保存したファイルをアップロードできます"
    )
    
    if uploaded_file is not None:
        try:
            # 📊 CSVファイルを読み込み
            df = pd.read_csv(uploaded_file)
            
            st.success(f"✅ ファイル '{uploaded_file.name}' を正常に読み込みました！")
            
            # 📋 データの基本情報を表示
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("📊 データ行数", len(df))
            with col2:
                st.metric("📈 列数", len(df.columns))
            with col3:
                st.metric("💾 データサイズ", f"{df.memory_usage(deep=True).sum() / 1024:.1f} KB")
            
            # 📄 データプレビュー
            st.subheader("📄 データプレビュー（最初の5行）")
            st.dataframe(df.head(), use_container_width=True)
            
            # 📋 詳細情報
            st.subheader("📋 データ詳細情報")
            col_info1, col_info2 = st.columns(2)
            
            with col_info1:
                st.markdown("**列名一覧:**")
                for i, col in enumerate(df.columns, 1):
                    st.write(f"{i}. {col}")
            
            with col_info2:
                st.markdown("**データ型:**")
                for col, dtype in df.dtypes.items():
                    st.write(f"- **{col}**: {dtype}")
            
            # 📊 数値列がある場合は簡単な統計を表示
            numeric_columns = df.select_dtypes(include=[np.number]).columns
            if len(numeric_columns) > 0:
                st.subheader("📊 数値列の基本統計")
                st.dataframe(df[numeric_columns].describe(), use_container_width=True)
                
        except Exception as e:
            st.error(f"❌ ファイルの読み込みでエラーが発生しました: {str(e)}")
            st.info("💡 ヒント: CSVファイルの文字エンコーディングがUTF-8であることを確認してください")
    
    else:
        st.info("📂 CSVファイルをアップロードすると、データの内容を確認できます")
        st.markdown("""
        **サンプルCSVファイルの作り方:**
        1. Excelやスプレッドシートでデータを作成
        2. 「名前を付けて保存」→「CSV (カンマ区切り)」を選択
        3. 作成したファイルをここにドラッグ&ドロップ
        """)
    
    st.markdown("---")
    
    # 🎲 ランダムデータ生成機能
    st.subheader("🎲 ランダムデータ生成")
    st.markdown("""
    **この機能について:**
    - プログラムで架空のテストデータを自動生成
    - データ分析の練習や機能テストに活用
    - 生成したデータはCSVファイルとしてダウンロード可能
    """)
    
    # 🎛️ データ生成の設定
    col_set1, col_set2 = st.columns(2)
    
    with col_set1:
        data_count = st.selectbox(
            "データ数を選択", 
            [10, 50, 100, 500, 1000],
            index=2,
            help="生成するデータの行数"
        )
    
    with col_set2:
        include_categories = st.checkbox(
            "カテゴリ列を追加", 
            value=True,
            help="A、B、Cなどのカテゴリデータを含めるかどうか"
        )
    
    # 🔄 データ生成ボタン
    if st.button("🎲 ランダムデータ生成", type="primary"):
        with st.spinner('データを生成中...'):
            # 🎯 基本データの生成
            random_df = pd.DataFrame({
                'ID': range(1, data_count + 1),
                '名前': [f'ユーザー{i:03d}' for i in range(1, data_count + 1)],
                'スコア': np.random.randint(0, 100, data_count),
                '年齢': np.random.randint(18, 65, data_count),
                '売上': np.random.normal(50000, 15000, data_count).astype(int),
                '日付': pd.date_range('2024-01-01', periods=data_count, freq='D')[:data_count]
            })
            
            # 🏷️ カテゴリ列の追加
            if include_categories:
                random_df['部署'] = np.random.choice(['営業', '開発', 'マーケティング', '人事'], data_count)
                random_df['地域'] = np.random.choice(['東京', '大阪', '名古屋', '福岡'], data_count)
            
            st.success(f"✅ {data_count}行のランダムデータを生成しました！")
            
            # 📊 生成データの表示
            st.dataframe(random_df, use_container_width=True)
            
            # 📊 簡単な統計情報
            col_stat1, col_stat2, col_stat3 = st.columns(3)
            with col_stat1:
                st.metric("平均スコア", f"{random_df['スコア'].mean():.1f}")
            with col_stat2:
                st.metric("平均年齢", f"{random_df['年齢'].mean():.1f}歳")
            with col_stat3:
                st.metric("平均売上", f"¥{random_df['売上'].mean():,.0f}")
            
            # 💾 CSVダウンロード機能
            csv_data = random_df.to_csv(index=False, encoding='utf-8-sig')  # Excelで正しく開ける文字コード
            
            st.download_button(
                label="📥 CSVファイルをダウンロード",
                data=csv_data,
                file_name=f'generated_data_{data_count}rows.csv',
                mime='text/csv',
                help="ダウンロードしたファイルはExcelやPythonで開くことができます"
            )
            
            st.caption("💡 ヒント: ダウンロードしたCSVファイルを上のアップロード機能で読み込むこともできます")

# 🎛️ サイドバーの説明と機能
st.sidebar.title("🎛️ サイドバー実験")
st.sidebar.markdown("""
**サイドバーの特徴:**
- メインコンテンツと分離された領域
- 設定やナビゲーションに使用
- 左側に常に表示される
""")

st.sidebar.info("💡 これは左側のサイドバーエリアです")

# 📊 プログレスバーのデモ
st.sidebar.subheader("📊 プログレスバーデモ")
if st.sidebar.button("進行状況を表示"):
    progress_bar = st.sidebar.progress(0)
    status_text = st.sidebar.empty()
    
    for i in range(101):
        progress_bar.progress(i)
        status_text.text(f'進行状況: {i}%')
        # 実際のアプリでは時間のかかる処理をここに書く
    
    st.sidebar.success("✅ 処理完了！")

st.sidebar.markdown("---")
st.sidebar.success("🧪 実験用アプリケーション起動中")

# 📚 学習リソースの紹介
st.sidebar.subheader("📚 学習リソース")
st.sidebar.markdown("""
**Streamlit公式:**
- [公式ドキュメント](https://docs.streamlit.io/)
- [チュートリアル](https://docs.streamlit.io/library/get-started)

**Python学習:**
- Pandas: データ操作
- NumPy: 数値計算  
- Plotly: グラフ作成
""")