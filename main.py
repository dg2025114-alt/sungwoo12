import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

# -----------------------------
# 페이지 기본 설정
# -----------------------------
st.set_page_config(page_title="서울 기온 예측기", layout="centered")
st.title("🌡️ 서울 연평균기온 예측기")

# -----------------------------
# 데이터 불러오기
# -----------------------------
DATA_URL = "https://raw.githubusercontent.com/greatsong/modudata/bb860932644270ad1199f10d3e7670e30231bce4/data/seoul.csv"

@st.cache_data
def load_data():
    df = pd.read_csv(DATA_URL, encoding="utf-8")
    df["날짜"] = pd.to_datetime(df["날짜"], errors="coerce")
    df = df.dropna(subset=["날짜", "평균기온"])
    df["연도"] = df["날짜"].dt.year
    return df

df = load_data()

# -----------------------------
# 연도별 집계: 평균기온과 관측일수
# -----------------------------
yearly = df.groupby("연도").agg(
    평균기온=("평균기온", "mean"),
    관측일수=("날짜", "count")
).reset_index()

# 조건: 2025년까지, 관측일 300일 이상
yearly_filtered = yearly[
    (yearly["연도"] <= 2025) & (yearly["관측일수"] >= 300)
].copy()

yearly_filtered = yearly_filtered.sort_values("연도").reset_index(drop=True)

n_years = len(yearly_filtered)
start_year = int(yearly_filtered["연도"].min())
end_year = int(yearly_filtered["연도"].max())

st.markdown(
    f"**회귀 직선 산출 기준**: 총 **{n_years}개년** 데이터 "
    f"({start_year}년 ~ {end_year}년, 2025년까지 & 관측일수 300일 이상)"
)

# -----------------------------
# 회귀 계수 계산 (1차 선형회귀)
# -----------------------------
x = yearly_filtered["연도"].values
y = yearly_filtered["평균기온"].values

slope, intercept = np.polyfit(x, y, 1)

# 상관계수
corr = np.corrcoef(x, y)[0, 1]

# -----------------------------
# 산점도 + 회귀직선 그래프
# -----------------------------
fig = go.Figure()

fig.add_trace(go.Scatter(
    x=x, y=y,
    mode="markers",
    name="연평균기온 (관측)",
    marker=dict(color="royalblue", size=8)
))

x_line = np.linspace(x.min(), x.max(), 100)
y_line = slope * x_line + intercept

fig.add_trace(go.Scatter(
    x=x_line, y=y_line,
    mode="lines",
    name="회귀 직선",
    line=dict(color="crimson", width=3)
))

fig.update_layout(
    title=f"서울 연평균기온 추세 (상관계수 r = {corr:.3f})",
    xaxis_title="연도",
    yaxis_title="평균기온 (°C)",
    hovermode="x unified"
)

st.plotly_chart(fig, use_container_width=True)

st.markdown(f"📈 **상관계수 (r)**: `{corr:.4f}`")
st.markdown(f"📐 **회귀식**: 평균기온 = {slope:.5f} × 연도 + ({intercept:.3f})")

# -----------------------------
# 슬라이더로 특정 연도 예측
# -----------------------------
st.divider()
st.subheader("🔍 연도별 예상 기온 확인하기")

selected_year = st.slider(
    "연도를 선택하세요",
    min_value=1900,
    max_value=2100,
    value=2025,
    step=1
)

predicted_temp = slope * selected_year + intercept

st.markdown(
    f"""
    <div style="text-align:center; padding: 30px; background-color:#f0f8ff; border-radius:15px;">
        <h3>{selected_year}년 예상 평균기온</h3>
        <h1 style="color:crimson; font-size:60px;">{predicted_temp:.2f} °C</h1>
    </div>
    """,
    unsafe_allow_html=True
)

if selected_year < start_year or selected_year > 2025:
    st.info("⚠️ 이 값은 회귀 직선을 이용한 추정치이며, 실제 관측 기준 범위를 벗어난 예측일 수 있습니다.")
