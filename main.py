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
# 회귀 계수 계산 (전체 기간)
# -----------------------------
x = yearly_filtered["연도"].values
y = yearly_filtered["평균기온"].values

slope, intercept = np.polyfit(x, y, 1)
corr = np.corrcoef(x, y)[0, 1]

# 100년당 상승폭
warming_per_100yr = slope * 100

# -----------------------------
# 최근 20년 데이터로 별도 회귀
# -----------------------------
recent_20 = yearly_filtered[yearly_filtered["연도"] >= (end_year - 19)].copy()

x_recent = recent_20["연도"].values
y_recent = recent_20["평균기온"].values

if len(recent_20) >= 2:
    slope_recent, intercept_recent = np.polyfit(x_recent, y_recent, 1)
    corr_recent = np.corrcoef(x_recent, y_recent)[0, 1]
    warming_per_100yr_recent = slope_recent * 100
else:
    slope_recent, intercept_recent, corr_recent, warming_per_100yr_recent = None, None, None, None

# -----------------------------
# 100년당 상승폭 크게 표시 (전체 vs 최근 20년 비교)
# -----------------------------
st.divider()
st.subheader("🔥 100년당 기온 상승폭 비교")

col1, col2 = st.columns(2)

with col1:
    st.markdown(
        f"""
        <div style="text-align:center; padding: 25px; background-color:#eef6ff; border-radius:15px;">
            <h4>전체 기간 ({start_year}~{end_year}년)</h4>
            <h1 style="color:royalblue; font-size:50px;">{warming_per_100yr:+.2f} °C</h1>
            <p>100년당 상승폭</p>
            <p>상관계수 r = {corr:.3f} · 자료 {n_years}개년</p>
        </div>
        """,
        unsafe_allow_html=True
    )

with col2:
    if warming_per_100yr_recent is not None:
        recent_start = int(recent_20["연도"].min())
        st.markdown(
            f"""
            <div style="text-align:center; padding: 25px; background-color:#fff0f0; border-radius:15px;">
                <h4>최근 20년 ({recent_start}~{end_year}년)</h4>
                <h1 style="color:crimson; font-size:50px;">{warming_per_100yr_recent:+.2f} °C</h1>
                <p>100년당 상승폭</p>
                <p>상관계수 r = {corr_recent:.3f} · 자료 {len(recent_20)}개년</p>
            </div>
            """,
            unsafe_allow_html=True
        )
    else:
        st.warning("최근 20년 데이터가 부족하여 계산할 수 없습니다.")

st.caption("※ '100년당 상승폭'은 회귀 직선의 기울기(연도당 상승량)에 100을 곱한 값입니다.")

# -----------------------------
# 산점도 + 회귀직선 그래프 (전체 vs 최근 20년 함께 표시)
# -----------------------------
st.divider()
st.subheader("📊 산점도 및 회귀 직선")

fig = go.Figure()

# 전체 관측치 산점도
fig.add_trace(go.Scatter(
    x=x, y=y,
    mode="markers",
    name="연평균기온 (관측)",
    marker=dict(color="royalblue", size=8)
))

# 전체 기간 회귀선
x_line = np.linspace(x.min(), x.max(), 100)
y_line = slope * x_line + intercept
fig.add_trace(go.Scatter(
    x=x_line, y=y_line,
    mode="lines",
    name=f"전체 기간 회귀선 ({warming_per_100yr:+.2f}°C/100년)",
    line=dict(color="royalblue", width=3)
))

# 최근 20년 회귀선
if slope_recent is not None:
    x_line_recent = np.linspace(x_recent.min(), x_recent.max(), 50)
    y_line_recent = slope_recent * x_line_recent + intercept_recent
    fig.add_trace(go.Scatter(
        x=x_line_recent, y=y_line_recent,
        mode="lines",
        name=f"최근 20년 회귀선 ({warming_per_100yr_recent:+.2f}°C/100년)",
        line=dict(color="crimson", width=3, dash="dash")
    ))

fig.update_layout(
    title="서울 연평균기온 추세",
    xaxis_title="연도",
    yaxis_title="평균기온 (°C)",
    hovermode="x unified"
)

st.plotly_chart(fig, use_container_width=True)

st.markdown(f"📈 **전체 기간 상관계수 (r)**: `{corr:.4f}`")
st.markdown(f"📐 **전체 기간 회귀식**: 평균기온 = {slope:.5f} × 연도 + ({intercept:.3f})")

# -----------------------------
# 슬라이더로 특정 연도 예측 (전체 기간 회귀식 기준)
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
        <h3>{selected_year}년 예상 평균기온 (전체 기간 회귀식 기준)</h3>
        <h1 style="color:crimson; font-size:60px;">{predicted_temp:.2f} °C</h1>
    </div>
    """,
    unsafe_allow_html=True
)

if selected_year < start_year or selected_year > 2025:
    st.info("⚠️ 이 값은 회귀 직선을 이용한 추정치이며, 실제 관측 기준 범위를 벗어난 예측일 수 있습니다.")
