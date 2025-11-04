# streamlit_app.py
# 코스피200 주식 추천 시스템 (KR UI)
# Data: Twelve Data API (API Key auth)
# NOTE: Educational demo only, NOT investment advice.

import os
import time
import math
from datetime import datetime
import numpy as np
import pandas as pd
import requests
import streamlit as st

# =========================
# Page Config
# =========================
st.set_page_config(
    page_title="코스피200 주식 추천 시스템",
    layout="wide",
    page_icon="📈"
)

# =========================
# Secrets & Config
# =========================
TD_API_KEY = st.secrets.get("twelvedata", {}).get("api_key") or os.getenv("TWELVEDATA_API_KEY", "")
TD_BASE = "https://api.twelvedata.com"

# Optional: broker-like creds for sidebar defaults
BROKER_APP_KEY = st.secrets.get("broker", {}).get("app_key", "")
BROKER_APP_SECRET = st.secrets.get("broker", {}).get("app_secret", "")
BROKER_ACCOUNT = st.secrets.get("broker", {}).get("account", "")

# =========================
# Sidebar — 설정
# =========================
with st.sidebar:
    st.header("설정", anchor=False)

    st.subheader("🔑 API 인증 정보", anchor=False)
    app_key = st.text_input("APP KEY", value=BROKER_APP_KEY, help="브로커/거래 API 연동 시 사용")
    app_secret = st.text_input("APP SECRET", value=BROKER_APP_SECRET, type="password", help="브로커/거래 API 연동 시 사용")
    account_no = st.text_input("계좌번호", value=BROKER_ACCOUNT, help="브로커/거래 API 연동 시 사용")

    # Twelve Data 상태
    if TD_API_KEY:
        st.success("Twelve Data API Key 감지됨")
    else:
        st.error("Twelve Data API Key가 없습니다. secrets.toml 또는 환경변수 TWELVEDATA_API_KEY 에 설정하세요.")

    st.subheader("📊 분석 설정", anchor=False)
    top_k = st.slider("추천받을 종목 개수", 3, 10, 5, 1)
    min_trading_ogwon = st.number_input("최소 거래 규모 (억원)", min_value=0, value=100, step=10,
                                        help="20일 평균 거래대금(억원) 필터")

    with st.expander("고급 설정 (클릭하여 펼치기)", expanded=False):
        universe_max = st.slider("분석 Universe 상한 (rate-limit 안전)", 10, 200, 60, 10)
        lookback_days = st.slider("지표 산출 기간 (days)", 60, 400, 200, 20)
        mom_window = st.slider("상승 속도 측정 기간 (days)", 10, 60, 20, 5)
        rsi_period = st.slider("RSI 기간", 7, 21, 14, 1)
        rsi_low, rsi_high = st.slider("적정 가격대 RSI 구간", 40, 80, (50, 70), 1)
        vol_window = st.slider("변동성(표준편차) 창 길이 (days)", 10, 60, 20, 5)
        max_vol_pct = st.number_input("허용 변동성 상한 (% / day stdev)", min_value=0.0, value=3.0, step=0.1,
                                      help="해당 상한 초과 시 패널티 또는 제외")
        include_high_vol = st.checkbox("변동성 상한 초과 종목도 포함하되 점수에서 패널티만 적용", value=True)

    st.markdown("---")
    st.markdown("**점수 가중치 (조절 가능)**")

    # Weights
    w_trend_enter = st.slider("✅ 상승 추세 진입 (+)", 0.0, 8.0, 4.0, 0.5,
                              help="가격>SMA50 & SMA20>SMA50 등 추세 성립 가중치")
    w_strong_up = st.slider("✅ 강한 상승세 (+)", 0.0, 5.0, 2.5, 0.5,
                            help=f"{mom_window}일 모멘텀·SMA크로스 강화 점수")
    w_volume_up = st.slider("✅ 거래 증가 (+)", 0.0, 3.0, 1.5, 0.5,
                            help="최근 거래대금이 20일 평균 대비 증가")
    w_fair_price = st.slider("✅ 적정 가격대 (+)", 0.0, 3.0, 1.5, 0.5,
                             help=f"RSI({rsi_period})가 [{rsi_low}–{rsi_high}] 구간")
    w_yesterday_up = st.slider("✅ 어제 대비 상승 (+)", 0.0, 2.0, 1.0, 0.5)
    w_high_vol_pen = st.slider("⚠️ 가격 변동 큼 (−)", 0.0, 2.0, 1.0, 0.5,
                               help=f"{vol_window}일 수익률 표준편차 기반 패널티 강도")

    start_btn = st.button("분석 시작하기", use_container_width=True, type="primary")

# =========================
# Header — Main Pane
# =========================
st.title("코스피200 주식 추천 시스템")
st.subheader("초보자도 쉽게 이해하는 주식 분석 도구")
st.info("👈 왼쪽 메뉴에서 API 정보를 입력하고 **'분석 시작하기'** 버튼을 눌러주세요!")

with st.expander("📌 이 도구는 무엇인가요?", expanded=True):
    st.write("코스피200 종목을 자동으로 분석하여 **매수하기 좋은 종목**을 추천해드립니다.")

with st.expander("분석 항목 (클릭하여 자세히 보기)", expanded=True):
    st.markdown("""
- 📈 **상승 추세**: 주가가 올라가는 흐름인지 확인 (가격>SMA50, SMA20>SMA50)
- 🚀 **상승 속도**: 최근 며칠간 얼마나 빠르게 올랐는지 (모멘텀)
- 💰 **거래 활발도**: 사람들이 얼마나 많이 거래하는지 (거래대금 증가)
- 📊 **적정 가격**: 너무 오르거나 떨어지지 않았는지 (RSI 밴드)
- ⚖️ **안정성**: 가격 변동이 크지 않은지 (단기 변동성)
""")

with st.expander("💯 추천 점수는 어떻게 계산하나요? (가중치/공식 보기)", expanded=False):
    st.markdown("""
| 항목 | 기본 점수(가중치 예시) |
|---|---|
| ✅ 상승 추세 진입 | +4점 |
| ✅ 강한 상승세 | +2~3점 |
| ✅ 거래 증가 | +1~2점 |
| ✅ 적정 가격대 | +1.5점 |
| ✅ 어제 대비 상승 | +1점 |
| ⚠️ 가격 변동 큼 | −0.5~−1점 |

**최종 점수 =** (상승 추세 가중치) + (상승 속도 가중치) + (거래 증가 가중치) + (RSI 밴드 가중치) + (어제 상승 가중치) − (변동성 패널티)

> 좌측 사이드바에서 각 가중치와 임계값을 자유롭게 조절할 수 있습니다.
""")

with st.expander("⚠️ 투자 주의사항", expanded=False):
    st.warning("이 도구는 참고용이며, 투자 손실에 대한 책임은 투자자 본인에게 있습니다. 실제 투자 전에는 반드시 추가 조사를 하시기 바랍니다.")

# =========================
# Utilities
# =========================
@st.cache_data(show_spinner=False)
def fetch_kospi200_symbols_from_wikipedia() -> pd.DataFrame:
    """Fetch KOSPI 200 (회사명 + 6자리 종목코드) from Wikipedia."""
    url = "https://en.wikipedia.org/wiki/KOSPI_200"
    tables = pd.read_html(url)
    candidates = [t for t in tables if {"Company", "Symbol"}.issubset(set(t.columns))]
    if not candidates:
        raise RuntimeError("KOSPI 200 구성표를 찾지 못했습니다.")
    df = candidates[0].copy()
    df["Symbol"] = df["Symbol"].astype(str).str.extract(r"(\d{6})", expand=False)
    df = df.dropna(subset=["Symbol"]).drop_duplicates(subset=["Symbol"]).reset_index(drop=True)
    df = df.rename(columns={"Company": "name", "Symbol": "symbol"})
    return df[["symbol", "name"]]

@st.cache_data(show_spinner=False)
def fallback_symbols() -> pd.DataFrame:
    data = [
        {"symbol": "005930", "name": "Samsung Electronics"},
        {"symbol": "000660", "name": "SK hynix"},
        {"symbol": "035420", "name": "NAVER"},
        {"symbol": "035720", "name": "Kakao"},
        {"symbol": "051910", "name": "LG Chem"},
        {"symbol": "005380", "name": "Hyundai Motor"},
        {"symbol": "207940", "name": "Samsung Biologics"},
        {"symbol": "068270", "name": "Celltrion"},
        {"symbol": "105560", "name": "KB Financial Group"},
        {"symbol": "096770", "name": "SK Innovation"},
    ]
    return pd.DataFrame(data)

def td_headers():
    return {"Authorization": f"apikey {TD_API_KEY}"} if TD_API_KEY else {}

def td_get(path: str, params: dict) -> dict:
    params = dict(params or {})
    if TD_API_KEY and "apikey" not in params:
        params["apikey"] = TD_API_KEY
    url = f"{TD_BASE.rstrip('/')}/{path.lstrip('/')}"
    r = requests.get(url, params=params, headers=td_headers(), timeout=30)
    r.raise_for_status()
    data = r.json()
    if isinstance(data, dict) and data.get("status") == "error":
        raise RuntimeError(f"Twelve Data error: {data.get('message')}")
    return data

@st.cache_data(show_spinner=False)
def fetch_daily_timeseries(symbol: str, out_size: int = 300) -> pd.DataFrame:
    """Daily OHLCV for 6-digit KRX code via Twelve Data time_series."""
    data = td_get("/time_series", {
        "symbol": symbol,
        "interval": "1day",
        "outputsize": out_size,
        "format": "JSON",
    })
    if "values" not in data:
        raise RuntimeError(f"No time_series values for {symbol}: {data}")
    df = pd.DataFrame(data["values"])
    df["datetime"] = pd.to_datetime(df["datetime"])
    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.sort_values("datetime").reset_index(drop=True)
    return df

def compute_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    roll_up = pd.Series(gain, index=series.index).rolling(period).mean()
    roll_down = pd.Series(loss, index=series.index).rolling(period).mean()
    rs = roll_up / (roll_down.replace(0, np.nan))
    rsi = 100 - (100 / (1 + rs))
    return rsi

def pct_change(a: float, b: float) -> float:
    if b == 0 or np.isnan(a) or np.isnan(b):
        return np.nan
    return (a / b - 1.0) * 100.0

def compute_indicators(ts: pd.DataFrame, lookback: int, mom_win: int, rsi_p: int, vol_win: int) -> dict:
    """Return last-day indicators & helper metrics."""
    if len(ts) < max(lookback, 60):
        return {}

    ts = ts.tail(lookback).copy()
    close = ts["close"]
    volume = ts["volume"]

    sma20 = close.rolling(20).mean()
    sma50 = close.rolling(50).mean()
    rsi = compute_rsi(close, rsi_p)
    vol20 = close.pct_change().rolling(vol_win).std() * 100  # % stdev

    latest = ts.iloc[-1]
    prev = ts.iloc[-2] if len(ts) >= 2 else latest
    latest_close = float(latest["close"])
    prev_close = float(prev["close"])
    latest_date = latest["datetime"]

    # Momentum over window
    if len(close) > mom_win:
        mom_ref = float(close.iloc[-(mom_win+1)])
        momentum_pct = pct_change(latest_close, mom_ref)
    else:
        momentum_pct = np.nan

    # Volume/Value
    value20 = (close * volume).rolling(20).mean().iloc[-1]
    avg20_ogwon = float(value20) / 1e8 if pd.notna(value20) else np.nan
    vol_last = float(vol20.iloc[-1]) if not np.isnan(vol20.iloc[-1]) else np.nan

    return {
        "date": latest_date.date().isoformat(),
        "close": latest_close,
        "prev_close": prev_close,
        "sma20": float(sma20.iloc[-1]),
        "sma50": float(sma50.iloc[-1]),
        "rsi": float(rsi.iloc[-1]),
        "momentum_pct": momentum_pct,
        "vol_stdev_pct": vol_last,
        "avg20_ogwon": avg20_ogwon,
    }

def score_row(ind: dict) -> dict:
    """Compute interpretable component scores."""
    if not ind:
        return {}

    s_trend_enter = 0.0
    if ind["close"] > ind["sma50"] and ind["sma20"] > ind["sma50"]:
        s_trend_enter = w_trend_enter

    s_strong_up = 0.0
    if not np.isnan(ind["momentum_pct"]):
        # Normalize momentum: +5% → +2.5점을 기준으로 선형 스케일링 (임의)
        s_strong_up = w_strong_up * (ind["momentum_pct"] / 5.0)
        s_strong_up = max(-w_strong_up, min(s_strong_up, w_strong_up))  # clamp

    s_volume_up = 0.0
    # 거래대금이 20일 평균 대비 의미 있게 증가했는지(직접증가율을 쓰기보다 최소 거래규모 만족을 보너스로)
    if not np.isnan(ind["avg20_ogwon"]) and ind["avg20_ogwon"] >= float(min_trading_ogwon):
        s_volume_up = w_volume_up

    s_fair_price = 0.0
    if rsi_low <= ind["rsi"] <= rsi_high:
        s_fair_price = w_fair_price
    elif ind["rsi"] > (rsi_high + 5):
        s_fair_price = -0.5  # 과매수 약한 패널티
    elif ind["rsi"] < (rsi_low - 5):
        s_fair_price = -0.5  # 과매도 약한 패널티

    s_yesterday_up = 0.0
    if ind["close"] > ind["prev_close"]:
        s_yesterday_up = w_yesterday_up

    s_vol_pen = 0.0
    if not np.isnan(ind["vol_stdev_pct"]):
        # 변동성 선형 패널티: 상한의 비율만큼 패널티 (초과 시 더 큰 패널티)
        ratio = ind["vol_stdev_pct"] / max(1e-9, float(max_vol_pct))
        s_vol_pen = -w_high_vol_pen * max(0.0, ratio - 1.0)  # 상한 이하일 때 0, 초과시 음수

    total = s_trend_enter + s_strong_up + s_volume_up + s_fair_price + s_yesterday_up + s_vol_pen

    return {
        "s_trend_enter": s_trend_enter,
        "s_strong_up": s_strong_up,
        "s_volume_up": s_volume_up,
        "s_fair_price": s_fair_price,
        "s_yesterday_up": s_yesterday_up,
        "s_vol_pen": s_vol_pen,
        "score": total
    }

# =========================
# Universe
# =========================
st.markdown("### Universe 로딩")
try:
    kospi_df = fetch_kospi200_symbols_from_wikipedia()
    st.success(f"KOSPI200 종목 {len(kospi_df)}개 불러옴")
except Exception as e:
    st.warning(f"자동 로딩 실패: {e} → 내장 샘플 사용")
    kospi_df = fallback_symbols()

universe = kospi_df.head(universe_max).copy()
st.dataframe(universe.head(20), use_container_width=True)

# =========================
# Run Analysis
# =========================
if not TD_API_KEY:
    st.stop()

if start_btn:
    out_size = max(lookback_days + 50, 180)
    rows = []
    progress = st.progress(0)
    status = st.empty()

    for i, r in universe.iterrows():
        sym = r["symbol"]
        name = r["name"]
        try:
            status.text(f"Fetching {sym} {name} ...")
            ts = fetch_daily_timeseries(sym, out_size=out_size)
            ind = compute_indicators(ts, lookback_days, mom_window, rsi_period, vol_window)
            if not ind:
                continue

            comp = score_row(ind)

            # 제외 로직: 변동성 상한 초과 & 포함 체크 해제
            if not include_high_vol and not np.isnan(ind["vol_stdev_pct"]) and ind["vol_stdev_pct"] > float(max_vol_pct):
                continue

            rows.append({
                "symbol": sym,
                "name": name,
                **ind,
                **comp
            })
        except Exception:
            time.sleep(0.2)  # API rate-limit 완화
        finally:
            progress.progress(int((i + 1) / len(universe) * 100))

    status.empty()

    if not rows:
        st.error("결과가 없습니다. Universe/기간/임계값을 조정해보세요.")
        st.stop()

    result = pd.DataFrame(rows)

    # Liquidity filter (again, in case min_trading_ogwon very high)
    liq_mask = result["avg20_ogwon"].fillna(0) >= float(min_trading_ogwon)
    passed = result[liq_mask].copy()
    if passed.empty:
        st.warning("최소 거래 규모 조건을 만족하는 종목이 없습니다. 조건을 낮추거나 Universe를 늘려보세요.")
        passed = result.copy()

    # Sort by final score
    passed = passed.sort_values("score", ascending=False).reset_index(drop=True)

    # KPI summary
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("분석 종목 수", len(result))
    with c2:
        st.metric("유동성 필터 통과", int(liq_mask.sum()))
    with c3:
        st.metric("상위 추천 수", top_k)
    with c4:
        st.metric("변동성 상한(%)", max_vol_pct)

    st.markdown("### 추천 결과")
    show_cols = [
        "symbol", "name", "date", "close",
        "momentum_pct", "rsi", "vol_stdev_pct", "avg20_ogwon",
        "s_trend_enter", "s_strong_up", "s_volume_up", "s_fair_price", "s_yesterday_up", "s_vol_pen",
        "score"
    ]
    st.dataframe(passed.loc[:, show_cols].head(top_k), use_container_width=True)

    # Download
    csv_bytes = passed.to_csv(index=False).encode("utf-8")
    st.download_button("CSV 다운로드", data=csv_bytes, file_name="kospi200_reco.csv", mime="text/csv")

    # Detail Cards (expanders)
    st.markdown("### 종목 상세 보기 (Top N)")
    detail_syms = passed["symbol"].head(top_k).tolist()

    for sym in detail_syms:
        row = passed[passed["symbol"] == sym].iloc[0]
        with st.expander(f"🔎 {row['symbol']} — {row['name']} (점수: {row['score']:.2f})", expanded=False):
            # Left: metrics, Right: chart
            lc, rc = st.columns([1, 2])

            with lc:
                st.markdown("**지표 요약**")
                st.write({
                    "날짜": row["date"],
                    "종가": round(float(row["close"]), 2),
                    "RSI": round(float(row["rsi"]), 2),
                    f"{mom_window}일 모멘텀(%)": None if np.isnan(row["momentum_pct"]) else round(float(row["momentum_pct"]), 2),
                    f"{vol_window}일 변동성 stdev(%)": None if np.isnan(row["vol_stdev_pct"]) else round(float(row["vol_stdev_pct"]), 2),
                    "20일 평균 거래대금(억원)": None if np.isnan(row["avg20_ogwon"]) else round(float(row["avg20_ogwon"]), 2),
                })

                st.markdown("**점수 구성**")
                st.write({
                    "상승 추세": round(float(row["s_trend_enter"]), 3),
                    "강한 상승세": round(float(row["s_strong_up"]), 3),
                    "거래 증가": round(float(row["s_volume_up"]), 3),
                    "적정 가격": round(float(row["s_fair_price"]), 3),
                    "어제 대비 상승": round(float(row["s_yesterday_up"]), 3),
                    "변동성 패널티": round(float(row["s_vol_pen"]), 3),
                    "총점": round(float(row["score"]), 3),
                })

            with rc:
                try:
                    ts_full = fetch_daily_timeseries(sym, out_size=max(lookback_days+80, 240))
                    ts = ts_full.tail(lookback_days).copy()
                    ts["SMA20"] = ts["close"].rolling(20).mean()
                    ts["SMA50"] = ts["close"].rolling(50).mean()

                    import altair as alt
                    base = alt.Chart(ts).encode(x="datetime:T")
                    price = base.mark_line().encode(y=alt.Y("close:Q", title="Price"))
                    sma20 = base.mark_line(strokeDash=[4,2]).encode(y="SMA20:Q")
                    sma50 = base.mark_line(strokeDash=[2,2]).encode(y="SMA50:Q")
                    st.altair_chart((price + sma20 + sma50).properties(height=320), use_container_width=True)

                    # Component bar chart
                    comp_df = pd.DataFrame({
                        "component": ["상승 추세", "강한 상승세", "거래 증가", "적정 가격", "어제 상승", "변동성 패널티"],
                        "value": [
                            row["s_trend_enter"], row["s_strong_up"], row["s_volume_up"],
                            row["s_fair_price"], row["s_yesterday_up"], row["s_vol_pen"]
                        ]
                    })
                    bar = alt.Chart(comp_df).mark_bar().encode(
                        x=alt.X("component:N", title="Component"),
                        y=alt.Y("value:Q", title="Score")
                    ).properties(height=220)
                    st.altair_chart(bar, use_container_width=True)

                except Exception as e:
                    st.warning(f"차트 표시 중 오류: {e}")

            st.markdown("---")
            st.caption("참고: 지표와 점수는 교육용/데모용이며 실제 투자 판단 근거로 사용하지 마세요.")

else:
    st.info("좌측에서 파라미터를 조정한 뒤 **분석 시작하기**를 눌러 결과를 확인하세요.")
