import streamlit as st
import duckdb
import pandas as pd
import altair as alt
from openai import OpenAI
import json
import re
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize OpenAI client
client = OpenAI()

# Security check for SQL
def is_safe_sql(sql):
    """Check if SQL query is safe (SELECT only)"""
    if not sql.strip().upper().startswith('SELECT'):
        return False
    
    forbidden_keywords = [
        'INSERT', 'UPDATE', 'DELETE', 'DROP', 'CREATE', 'ALTER', 
        'TRUNCATE', 'GRANT', 'REVOKE', 'EXEC', 'EXECUTE', 'CALL'
    ]
    
    sql_upper = sql.upper()
    for keyword in forbidden_keywords:
        if keyword in sql_upper:
            return False
    
    return True

# Generate SQL using LLM
def generate_sql(question, table_schema):
    """Generate SQL query from natural language question"""
    prompt = f"""あなたはDuckDBのSQLエキスパートです。自然文の質問からSELECT文のみを含むJSONを生成してください。

テーブル: sales
スキーマ: {table_schema}

ルール:
- SELECT文のみ（DDL/DML禁止）
- 売上は SUM(revenue) を基本とする
- 月次集計は CAST(date_trunc('month', date) AS DATE) AS month
- 文字列リテラルは必ずシングルクォート('文字列')を使用（ダブルクォート禁止）
- 例: WHERE region = 'North' （正しい）
- 例: WHERE region = "North" （間違い）
- 出力は必ずJSONのみ

JSONスキーマ:
{{
  "sql": "<SELECT only>",
  "chart": "<bar|line|area|table>",
  "x": "<X軸列名 or null>",
  "y": "<Y軸列名 or null>",
  "series": "<系列列名 or null>",
  "note": "<短い補足>"
}}

質問: {question}"""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )
    
    try:
        result = json.loads(response.choices[0].message.content.strip())
        return result
    except json.JSONDecodeError:
        # Try to extract JSON from response if it's wrapped in other text
        content = response.choices[0].message.content
        json_match = re.search(r'\{.*\}', content, re.DOTALL)
        if json_match:
            return json.loads(json_match.group())
        raise ValueError("Invalid JSON response from LLM")

# Fix SQL using LLM
def fix_sql(failed_sql, error_message, table_schema):
    """Fix failed SQL using LLM"""
    prompt = f"""以下のSQLでエラーが発生しました。修正版のJSONを返してください。

失敗したSQL: {failed_sql}
エラー: {error_message}
テーブルスキーマ: {table_schema}

修正時の重要なルール:
- 文字列リテラルは必ずシングルクォート('文字列')を使用
- ダブルクォートは絶対に使用しない
- DuckDB標準のSQL構文に従う

同じJSONスキーマで修正版を返してください:
{{
  "sql": "<修正されたSELECT>",
  "chart": "<bar|line|area|table>",
  "x": "<X軸列名 or null>",
  "y": "<Y軸列名 or null>", 
  "series": "<系列列名 or null>",
  "note": "<短い補足>"
}}"""

    response = client.chat.completions.create(
        model="gpt-4o-mini", 
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )
    
    try:
        result = json.loads(response.choices[0].message.content.strip())
        return result
    except json.JSONDecodeError:
        content = response.choices[0].message.content
        json_match = re.search(r'\{.*\}', content, re.DOTALL)
        if json_match:
            return json.loads(json_match.group())
        raise ValueError("Invalid JSON response from LLM")

# Generate summary using LLM
def generate_summary(question, result_data, note, sql_query):
    """Generate Japanese summary of results"""
    prompt = f"""質問: {question}
実行SQL: {sql_query}
結果（先頭5行）:
{result_data.head().to_string() if len(result_data) > 0 else "データなし"}
補足: {note}

上記の結果について、日本語で詳細な要約を作成してください。
以下の形式で記述：

## 📊 計算方法
- 使用したSQL式と計算ロジックを具体的に説明
- 例：「売上合計はSUM(revenue)で算出」「平均単価はSUM(revenue)/SUM(units)で計算」

## 🔍 詳細分析
- データから読み取れる数値的な特徴（最大値、最小値、平均値など）
- 上位・下位項目の具体的な数値と割合
- 時系列の場合は増減率や変化パターン
- 比較分析の場合は具体的な差異と倍率

## 💡 ビジネスインサイト
- データから導かれる具体的な示唆
- 注目すべきトレンドやパターン
- 改善や戦略のためのアクションポイント

## 📋 結論
- 最も重要な発見を1-2文で要約

数値は具体的に、パーセンテージや比率も含めて詳しく分析してください。"""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3
    )
    
    return response.choices[0].message.content.strip()

# Create chart using Altair
def create_chart(df, chart_type, x_col, y_col, series_col):
    """Create chart using Altair based on specifications"""
    if chart_type == "table" or x_col is None or y_col is None:
        return None
        
    try:
        base = alt.Chart(df)
        
        if series_col and series_col in df.columns:
            color_encoding = alt.Color(f'{series_col}:N')
        else:
            color_encoding = alt.value('steelblue')
            
        tooltip = list(df.columns)
        
        if chart_type == "bar":
            chart = base.mark_bar().encode(
                x=alt.X(f'{x_col}:O' if df[x_col].dtype == 'object' else f'{x_col}:Q'),
                y=alt.Y(f'{y_col}:Q'),
                color=color_encoding,
                tooltip=tooltip
            )
        elif chart_type == "line":
            chart = base.mark_line(point=True).encode(
                x=alt.X(f'{x_col}:T' if 'date' in x_col.lower() or 'month' in x_col.lower() else f'{x_col}:O'),
                y=alt.Y(f'{y_col}:Q'),
                color=color_encoding,
                tooltip=tooltip
            )
        elif chart_type == "area":
            chart = base.mark_area(opacity=0.7).encode(
                x=alt.X(f'{x_col}:T' if 'date' in x_col.lower() or 'month' in x_col.lower() else f'{x_col}:O'),
                y=alt.Y(f'{y_col}:Q'),
                color=color_encoding,
                tooltip=tooltip
            )
        else:
            return None
            
        return chart.resolve_scale(color='independent').properties(
            width=600,
            height=400
        )
    except Exception as e:
        st.error(f"チャート作成エラー: {e}")
        return None

# Load and setup data
@st.cache_resource
def setup_database():
    """Setup DuckDB with sales data"""
    conn = duckdb.connect(":memory:")
    
    # Load CSV data
    data_path = "data/sample_sales.csv"
    if not os.path.exists(data_path):
        st.error("データファイルが見つかりません: data/sample_sales.csv")
        return None, None
        
    df = pd.read_csv(data_path)
    
    # Convert date column
    df['date'] = pd.to_datetime(df['date'])
    
    # Create table in DuckDB
    conn.register('sales', df)
    
    return conn, df

# Quick analysis functions
def quick_analysis_monthly_category():
    """Monthly sales by category"""
    return "月毎のカテゴリー別の売り上げを教えて"

def quick_analysis_channel():
    """Sales by channel"""
    return "チャネルごとの売り上げを教えて"

def quick_analysis_region():
    """Sales by region"""  
    return "地域ごとの売り上げの合計を教えて"

# Additional analysis suggestions
def get_all_analysis_suggestions():
    """Get full list of analysis suggestions"""
    return [
        "2025年1月の地域×チャネル別の売上",
        "顧客セグメント別の平均単価", 
        "カテゴリ別の販売数量ランキング",
        "日別の売上推移トレンド",
        "最も収益性の高い商品カテゴリ",
        "オンラインとStore売上の比較",
        "週末と平日の売上パターン比較",
        "単価が最も高いカテゴリの分析",
        "Small Businessセグメントの購買傾向", 
        "月初と月末の売上動向",
        "北部地域のカテゴリ別売上構成",
        "Corporate顧客の購買特性",
        "販売チャネル別の平均取引額",
        "Beauty商品の地域別売上分布",
        "Electronics売上の時系列変動",
        "顧客セグメント×地域の売上マトリクス"
    ]

def get_analysis_suggestions(count=6):
    """Get random analysis suggestions"""
    import random
    all_suggestions = get_all_analysis_suggestions()
    return random.sample(all_suggestions, min(count, len(all_suggestions)))

# Main app
def main():
    st.set_page_config(
        page_title="売上データ分析AIチャットボット",
        page_icon="📊",
        layout="wide"
    )
    
    st.title("売上データ分析AIチャットボット（DuckDB＋SQL自動生成）")
    
    # Setup database
    conn, df = setup_database()
    if conn is None:
        st.stop()
    
    # Get table schema
    table_info = conn.execute("DESCRIBE sales").fetchall()
    schema_str = ", ".join([f"{col[0]} ({col[1]})" for col in table_info])
    
    # Sidebar
    with st.sidebar:
        st.header("📈 データ概要")
        
        try:
            total_records = len(df)
            date_range = f"{df['date'].min().strftime('%Y-%m-%d')} ～ {df['date'].max().strftime('%Y-%m-%d')}"
            categories = ", ".join(df['category'].unique()[:3])
            if len(df['category'].unique()) > 3:
                categories += "..."
                
            regions = ", ".join(df['region'].unique())
            channels = ", ".join(df['sales_channel'].unique())
            
            st.metric("総レコード数", f"{total_records:,}")
            st.write(f"**期間**: {date_range}")
            st.write(f"**カテゴリ**: {categories}")  
            st.write(f"**地域**: {regions}")
            st.write(f"**チャネル**: {channels}")
            
        except Exception as e:
            st.error(f"データ概要の取得に失敗: {e}")
        
        st.divider()
        
        st.header("🚀 クイック分析")
        
        if st.button("月×カテゴリの売上推移", use_container_width=True):
            st.session_state.quick_question = quick_analysis_monthly_category()
            
        if st.button("チャネル別の売上合計", use_container_width=True):
            st.session_state.quick_question = quick_analysis_channel()
            
        if st.button("地域別の売上合計", use_container_width=True):
            st.session_state.quick_question = quick_analysis_region()
            
        st.divider()
        
        # Header with reload button
        col1, col2 = st.columns([3, 1])
        with col1:
            st.header("💡 次の分析提案")
        with col2:
            if st.button("🔄", help="新しい提案を表示", key="reload_suggestions"):
                if "suggestion_seed" in st.session_state:
                    st.session_state.suggestion_seed += 1
                else:
                    st.session_state.suggestion_seed = 1
                st.rerun()
        
        # Initialize seed for consistent suggestions until reload
        if "suggestion_seed" not in st.session_state:
            st.session_state.suggestion_seed = 0
            
        # Set random seed for consistent suggestions
        import random
        random.seed(st.session_state.suggestion_seed)
        
        suggestions = get_analysis_suggestions()
        for i, suggestion in enumerate(suggestions):
            if st.button(suggestion, key=f"suggest_{st.session_state.suggestion_seed}_{i}", use_container_width=True):
                st.session_state.quick_question = suggestion
    
    # Main chat interface
    if "messages" not in st.session_state:
        st.session_state.messages = []
        
    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            if message["role"] == "assistant":
                if "sql" in message:
                    st.caption(f"実行SQL: {message['sql']}")
                if "dataframe" in message:
                    st.subheader("📊 集計結果")
                    st.dataframe(message["dataframe"], use_container_width=True)
                if "chart" in message and message["chart"]:
                    st.subheader("📈 グラフ")
                    st.altair_chart(message["chart"], use_container_width=True)
                if "summary" in message:
                    st.subheader("💡 要約")
                    st.write(message["summary"])
            else:
                st.write(message["content"])
    
    # Chat input (always display)
    user_input = st.chat_input("売上データについて質問してください...")
    
    # Handle quick question or user input
    question = None
    if hasattr(st.session_state, 'quick_question'):
        question = st.session_state.quick_question
        del st.session_state.quick_question
    elif user_input:
        question = user_input
        
    if question:
        # Add user message
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.write(question)
            
        # Generate response
        with st.chat_message("assistant"):
            try:
                # Generate SQL
                with st.spinner("SQLを生成中..."):
                    sql_spec = generate_sql(question, schema_str)
                
                sql_query = sql_spec.get("sql", "")
                
                # Safety check
                if not is_safe_sql(sql_query):
                    st.error("安全でないSQLが検出されました。SELECT文のみ使用可能です。")
                    st.stop()
                
                st.caption(f"実行SQL: {sql_query}")
                
                # Execute SQL
                try:
                    result_df = conn.execute(sql_query).fetch_df()
                except Exception as e:
                    st.warning(f"SQL実行エラー: {e}")
                    st.info("SQLを自動修正しています...")
                    
                    try:
                        # Try to fix SQL
                        fixed_spec = fix_sql(sql_query, str(e), schema_str)
                        fixed_sql = fixed_spec.get("sql", "")
                        
                        if not is_safe_sql(fixed_sql):
                            st.error("修正後のSQLも安全ではありません。")
                            st.stop()
                            
                        st.caption(f"修正後SQL: {fixed_sql}")
                        result_df = conn.execute(fixed_sql).fetch_df()
                        sql_spec = fixed_spec
                        
                    except Exception as fix_error:
                        st.error(f"SQL修正も失敗しました: {fix_error}")
                        st.stop()
                
                # Display results
                if len(result_df) == 0:
                    st.warning("該当するデータが見つかりませんでした。")
                    st.stop()
                
                st.subheader("📊 集計結果")
                st.dataframe(result_df, use_container_width=True)
                
                # Create chart
                chart = None
                if sql_spec.get("chart") != "table":
                    chart = create_chart(
                        result_df, 
                        sql_spec.get("chart"),
                        sql_spec.get("x"),
                        sql_spec.get("y"),
                        sql_spec.get("series")
                    )
                    
                    if chart:
                        st.subheader("📈 グラフ") 
                        st.altair_chart(chart, use_container_width=True)
                
                # Generate summary
                with st.spinner("要約を生成中..."):
                    summary = generate_summary(
                        question,
                        result_df, 
                        sql_spec.get("note", ""),
                        sql_query
                    )
                
                st.subheader("💡 要約")
                st.write(summary)
                
                # Save to chat history
                assistant_msg = {
                    "role": "assistant",
                    "content": "分析完了",
                    "sql": sql_query,
                    "dataframe": result_df,
                    "chart": chart,
                    "summary": summary
                }
                st.session_state.messages.append(assistant_msg)
                
            except Exception as e:
                st.error(f"分析中にエラーが発生しました: {e}")

if __name__ == "__main__":
    main()