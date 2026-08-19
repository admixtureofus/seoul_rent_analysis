# -*- coding: utf-8 -*-
"""
========================================================================
서울시 전월세가 데이터 전처리 + EDA/시각화
------------------------------------------------------------------------
대상 데이터 : 서울시 부동산 전월세가 정보 (data.seoul.go.kr, OA-21276)
목적        : 2030 사회초년생 대상 '동네 추천 대시보드' - 월세 축 분석
분석 프레임 : 가격 / 지역 / 시간 / 주택유형 + '싼 동네 vs 가성비 동네' 비교

※ 실제 CSV의 헤더명이 아래 COLMAP과 다르면, COLMAP 우측 값만
   실제 컬럼명으로 바꿔주면 스크립트 전체가 그대로 동작합니다.
========================================================================
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import seaborn as sns

# ------------------------------------------------------------------
# 0. 환경 설정 (한글 폰트, 스타일)
# ------------------------------------------------------------------
def set_korean_font():
    candidates = ["NanumGothic", "NanumBarunGothic", "Malgun Gothic", "AppleGothic"]
    installed = {f.name for f in fm.fontManager.ttflist}
    if not any(c in installed for c in candidates):
        # 폰트가 설치 직후라 matplotlib 캐시에 안 잡혀 있는 경우 강제 재빌드
        try:
            fm._load_fontmanager(try_read_cache=False)
            installed = {f.name for f in fm.fontManager.ttflist}
        except Exception:
            pass
    for name in candidates:
        if name in installed:
            plt.rcParams["font.family"] = "sans-serif"
            plt.rcParams["font.sans-serif"] = [name] + plt.rcParams["font.sans-serif"]
            break
    else:
        print("[경고] 한글 폰트를 찾지 못했습니다. (리눅스: apt-get install -y fonts-nanum 후 재실행 권장)")
    plt.rcParams["axes.unicode_minus"] = False

sns.set_style("whitegrid")  # 폰트 설정보다 먼저 호출 (seaborn이 font.family를 초기화하므로)
set_korean_font()
plt.rcParams["figure.dpi"] = 110

INPUT_PATH = r"C:\Users\과표사업단\Downloads\싱글벙글\서울특별시_전월세가"          # 원본 데이터 경로 (raw CSV 또는 폴더 내 여러 연도 파일 병합)
OUTPUT_DIR = "output"                  # 정제 데이터 / 그래프 저장 폴더
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ------------------------------------------------------------------
# 1. 컬럼명 매핑 
# ------------------------------------------------------------------
COLMAP = {
    "gu":        "자치구명",      # 자치구
    "dong":      "법정동명",      # 법정동
    "type":      "전월세구분",     # '전세' / '월세'
    "housing":   "건물용도",      # 아파트/연립다세대/오피스텔 등
    "deposit":   "보증금(만원)",   # 보증금
    "rent":      "임대료(만원)",   # 월세
    "area":      "임대면적",   # 전용/임대면적
    "floor":     "층",
    "build_year":"건축년도",
    "year":      "접수년도",      # 계약(접수)연도
}

# 서울시 자치구 발표 기준 전월세전환율(연, %)
CONVERSION_RATE = 4.5  # % (서울시 평균 수준 참고값 -> 법정 상한 기준)


# ========================================================================
# 2. 데이터 로드
# ========================================================================
def load_data(path=INPUT_PATH):
    """
    단일 CSV 또는 연도별로 나뉜 여러 CSV를 하나로 병합해서 로드.
    - 폴더 경로가 들어오면 폴더 내 모든 csv를 concat
    - 인코딩은 cp949 / utf-8 순서로 시도
    """
    def _read_one(fp):
        for enc in ["utf-8", "cp949", "euc-kr"]:
            try:
                return pd.read_csv(fp, encoding=enc, low_memory=False)
            except UnicodeDecodeError:
                continue
        raise UnicodeDecodeError(f"인코딩 판별 실패: {fp}")

    if os.path.isdir(path):
        files = [os.path.join(path, f) for f in os.listdir(path) if f.endswith(".csv")]
        dfs = [_read_one(f) for f in files]
        df = pd.concat(dfs, ignore_index=True)
    else:
        df = _read_one(path)

    print(f"[로드 완료] shape={df.shape}")
    return df


# ========================================================================
# 3. 전처리
# ========================================================================
def preprocess(df: pd.DataFrame, include_semi_jeonse=True) -> pd.DataFrame:
    """
    1) 컬럼명 표준화
    2) 월세 데이터만 필터링 (전세 제외)
    3) 결측치/이상치 제거
    4) 파생변수 생성: 환산월세, 면적당월세, 주택규모(원룸급 등), 층분류
    """
    c = COLMAP
    df = df.rename(columns={v: k for k, v in c.items()})

    # --- 3-1. 숫자형 변환 (콤마, 공백 등 정제) ---
    for col in ["deposit", "rent", "area", "floor", "build_year", "year"]:
        if col in df.columns:
            df[col] = (
                df[col]
                .astype(str)
                .str.replace(",", "", regex=False)
                .str.strip()
                .replace({"": np.nan, "nan": np.nan})
            )
            df[col] = pd.to_numeric(df[col], errors="coerce")

    n0 = len(df)

    # --- 3-2. 전월세구분 필터링 : 월세만 사용 ---
    # 데이터 값이 '월세'/'준월세'/'준전세' 등으로 세분화되어 있을 수 있어 '전세'만 명시적으로 제외
    if "type" in df.columns:
        df = df[~df["type"].astype(str).str.contains("전세", na=False)]
    print(f"[필터] 전세 제외 -> {n0} -> {len(df)}건")

    # 반전세(보증금이 매우 큰 월세) 포함 여부 옵션
    # include_semi_jeonse=False 이면 보증금이 지나치게 큰 반전세성 계약 제외
    if not include_semi_jeonse and "deposit" in df.columns:
        before = len(df)
        # 보증금이 억 단위(예: 10000만원=1억) 이상인 반전세성 계약 제외 (임계값은 목적에 맞게 조정)
        df = df[df["deposit"] < 10000]
        print(f"[필터] 반전세 제외 옵션 적용 -> {before} -> {len(df)}건")

    # --- 3-3. 결측치 제거 (핵심 변수 기준) ---
    core_cols = [c for c in ["rent", "area", "gu"] if c in df.columns]
    n1 = len(df)
    df = df.dropna(subset=core_cols)
    print(f"[정제] 핵심변수 결측 제거 -> {n1} -> {len(df)}건")

    # --- 3-4. 이상치 제거 ---
    n2 = len(df)
    df = df[(df["rent"] > 0) & (df["rent"] < 1000)]        # 월세 0 이하 / 비정상 고가(예: 1000만원 이상) 제외
    df = df[(df["area"] > 5) & (df["area"] < 300)]          # 면적 5㎡ 미만, 300㎡ 이상 제외
    if "deposit" in df.columns:
        df = df[(df["deposit"] >= 0)]
    if "floor" in df.columns:
        df = df[(df["floor"].isna()) | ((df["floor"] > -3) & (df["floor"] < 70))]
    print(f"[정제] 이상치 제거 -> {n2} -> {len(df)}건")

    # IQR 기반 추가 극단치 제거 (면적당 월세 계산 전, 월세 기준)
    q1, q3 = df["rent"].quantile([0.25, 0.75])
    iqr = q3 - q1
    lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    n3 = len(df)
    df = df[(df["rent"] >= lower) & (df["rent"] <= upper)]
    print(f"[정제] IQR 극단치 제거 -> {n3} -> {len(df)}건 (기준: {lower:.1f}~{upper:.1f}만원)")

    # --- 3-5. 파생변수 ---
    # 환산월세 = 월세 + 보증금 * (전환율/12/100)   [단위: 만원]
    if "deposit" in df.columns:
        df["rent_converted"] = df["rent"] + df["deposit"] * (CONVERSION_RATE / 12 / 100)
    else:
        df["rent_converted"] = df["rent"]

    # 면적당 월세 (만원/㎡) - 원 월세 기준, 환산월세 기준 둘 다 생성
    df["rent_per_area"] = df["rent"] / df["area"]
    df["rent_conv_per_area"] = df["rent_converted"] / df["area"]

    # 방 규모 추정 (전용면적 기준 러프 구간 - 사회초년생向 직관적 라벨)
    bins = [0, 20, 33, 60, np.inf]
    labels = ["원룸급(~20㎡)", "투룸급(20~33㎡)", "중형(33~60㎡)", "대형(60㎡~)"]
    df["room_size"] = pd.cut(df["area"], bins=bins, labels=labels)

    # 건축연차 (분석 시점 기준 신축/구축 구분용)
    if "build_year" in df.columns and "year" in df.columns:
        df["building_age"] = df["year"] - df["build_year"]

    # 결측 주택유형 처리
    if "housing" in df.columns:
        df["housing"] = df["housing"].fillna("기타")

    df = df.reset_index(drop=True)
    print(f"[전처리 완료] 최종 {len(df)}건, 컬럼: {list(df.columns)}")
    return df


# ========================================================================
# 4. 요약 통계 (자치구 / 법정동 단위) - 표본수 부족 지역 플래그 포함
# ========================================================================
def summarize_by_region(df: pd.DataFrame, group_col: str, min_n=30) -> pd.DataFrame:
    agg = df.groupby(group_col).agg(
        n=("rent", "size"),
        rent_mean=("rent", "mean"),
        rent_median=("rent", "median"),
        rent_conv_mean=("rent_converted", "mean"),
        deposit_mean=("deposit", "mean"),
        deposit_median=("deposit", "median"),
        area_mean=("area", "mean"),
        rent_per_area_mean=("rent_per_area", "mean"),
        rent_per_area_median=("rent_per_area", "median"),
    ).reset_index()

    agg["reliable"] = agg["n"] >= min_n  # 표본 30건 미만 = 참고용으로만 사용 권장
    agg = agg.sort_values("rent_per_area_mean")
    return agg


# ========================================================================
# 5. 가성비 스코어 (자치구 평균 대비 상대값으로 표준화) -- 가성비라는 표현이 주관적이긴함
# ========================================================================
def compute_value_score(gu_summary: pd.DataFrame) -> pd.DataFrame:
    """
    '싼 동네'(rent_mean 낮음)와 '가성비 동네'(rent_per_area_mean 낮음)를
    각각 순위화하고, 두 순위의 차이를 비교할 수 있게 함.
    """
    df = gu_summary.copy()
    df["rank_cheap"] = df["rent_mean"].rank(method="min")             # 절대 월세 기준 순위
    df["rank_value"] = df["rent_per_area_mean"].rank(method="min")    # 면적당 월세(가성비) 기준 순위
    df["rank_gap"] = df["rank_cheap"] - df["rank_value"]
    # rank_gap > 0 : 절대월세는 비싸보였지만 실제 면적 대비로는 저렴한(=가성비 좋은) 지역
    # rank_gap < 0 : 절대월세는 싸보였지만 면적이 작아 오히려 면적당으로는 비싼 지역
    return df.sort_values("rank_value")


# ========================================================================
# 6. 시각화
# ========================================================================
def plot_price_distribution(df, save=True):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    sns.histplot(df["rent"], bins=50, kde=True, ax=axes[0], color="#4C72B0")
    axes[0].axvline(df["rent"].mean(), color="red", ls="--", label=f"평균 {df['rent'].mean():.0f}만원")
    axes[0].axvline(df["rent"].median(), color="green", ls="--", label=f"중앙값 {df['rent'].median():.0f}만원")
    axes[0].set_title("월세 분포")
    axes[0].set_xlabel("월세(만원)")
    axes[0].legend()

    sns.histplot(df["rent_per_area"], bins=50, kde=True, ax=axes[1], color="#DD8452")
    axes[1].axvline(df["rent_per_area"].mean(), color="red", ls="--",
                     label=f"평균 {df['rent_per_area'].mean():.2f}")
    axes[1].axvline(df["rent_per_area"].median(), color="green", ls="--",
                     label=f"중앙값 {df['rent_per_area'].median():.2f}")
    axes[1].set_title("면적당 월세 분포 (만원/㎡)")
    axes[1].set_xlabel("만원/㎡")
    axes[1].legend()

    plt.tight_layout()
    if save:
        plt.savefig(f"{OUTPUT_DIR}/01_price_distribution.png")
    plt.show()


def plot_by_gu(gu_summary, save=True):
    top20 = gu_summary.sort_values("rent_mean").head(25)  # 서울 25개구 전체
    fig, axes = plt.subplots(1, 2, figsize=(14, 8))

    order = gu_summary.sort_values("rent_mean")["gu"]
    sns.barplot(data=gu_summary, x="rent_mean", y="gu", order=order, ax=axes[0], color="#4C72B0")
    axes[0].set_title("자치구별 평균 월세")
    axes[0].set_xlabel("평균 월세(만원)")

    order2 = gu_summary.sort_values("rent_per_area_mean")["gu"]
    sns.barplot(data=gu_summary, x="rent_per_area_mean", y="gu", order=order2, ax=axes[1], color="#DD8452")
    axes[1].set_title("자치구별 면적당 월세 (가성비)")
    axes[1].set_xlabel("만원/㎡")

    plt.tight_layout()
    if save:
        plt.savefig(f"{OUTPUT_DIR}/02_by_gu.png")
    plt.show()


def plot_cheap_vs_value(gu_score, save=True, top_n=25):
    """'싼 동네' 순위와 '가성비 동네' 순위 차이를 한눈에 보여주는 슬로프 차트"""
    df = gu_score.sort_values("rank_value").head(top_n)
    fig, ax = plt.subplots(figsize=(7, 9))

    for _, row in df.iterrows():
        ax.plot([0, 1], [row["rank_cheap"], row["rank_value"]], color="gray", alpha=0.5, zorder=1)

    ax.scatter([0] * len(df), df["rank_cheap"], color="#4C72B0", zorder=2, label="절대 월세 순위")
    ax.scatter([1] * len(df), df["rank_value"], color="#DD8452", zorder=2, label="면적당 월세(가성비) 순위")

    for _, row in df.iterrows():
        ax.text(-0.05, row["rank_cheap"], row["gu"], ha="right", va="center", fontsize=9)
        ax.text(1.05, row["rank_value"], row["gu"], ha="left", va="center", fontsize=9)

    ax.set_xticks([0, 1])
    ax.set_xticklabels(["싼 동네 순위\n(절대 월세)", "가성비 동네 순위\n(면적당 월세)"])
    ax.invert_yaxis()
    ax.set_ylabel("순위 (1위 = 가장 저렴)")
    ax.set_title("'월세가 싼 동네' vs '가성비 좋은 동네' 비교")
    plt.tight_layout()
    if save:
        plt.savefig(f"{OUTPUT_DIR}/03_cheap_vs_value.png")
    plt.show()


def plot_yearly_trend(df, save=True):
    yearly = df.groupby("year").agg(
        rent_mean=("rent", "mean"),
        rent_median=("rent", "median"),
        rent_per_area_mean=("rent_per_area", "mean"),
    ).reset_index()

    yearly["rent_yoy"] = yearly["rent_mean"].pct_change() * 100

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    axes[0].plot(yearly["year"], yearly["rent_mean"], marker="o", label="평균")
    axes[0].plot(yearly["year"], yearly["rent_median"], marker="o", label="중앙값")
    axes[0].set_title("연도별 월세 추이")
    axes[0].set_xlabel("연도")
    axes[0].set_ylabel("월세(만원)")
    axes[0].legend()

    recent = yearly[yearly["year"] >= yearly["year"].max() - 4]
    sns.barplot(data=recent, x="year", y="rent_yoy", ax=axes[1], color="#C44E52")
    axes[1].axhline(0, color="black", lw=0.8)
    axes[1].set_title("최근 5개년 전년 대비 상승률(%)")
    axes[1].set_ylabel("상승률(%)")

    plt.tight_layout()
    if save:
        plt.savefig(f"{OUTPUT_DIR}/04_yearly_trend.png")
    plt.show()
    return yearly


def plot_by_housing_type(df, save=True):
    if "housing" not in df.columns:
        print("housing 컬럼 없음 - 스킵")
        return
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    sns.boxplot(data=df, x="housing", y="rent", ax=axes[0])
    axes[0].set_title("주택유형별 월세 분포")
    axes[0].tick_params(axis="x", rotation=20)

    sns.boxplot(data=df, x="housing", y="rent_per_area", ax=axes[1])
    axes[1].set_title("주택유형별 면적당 월세 분포")
    axes[1].tick_params(axis="x", rotation=20)

    plt.tight_layout()
    if save:
        plt.savefig(f"{OUTPUT_DIR}/05_by_housing_type.png")
    plt.show()


def plot_rent_vs_area_scatter(df, save=True, sample=5000):
    """월세 vs 면적 산점도 - A동/B동 같은 케이스를 직관적으로 확인"""
    d = df.sample(min(sample, len(df)), random_state=42)
    fig, ax = plt.subplots(figsize=(8, 6))
    scatter = ax.scatter(d["area"], d["rent"], c=d["rent_per_area"], cmap="viridis_r", alpha=0.5, s=15)
    cb = plt.colorbar(scatter)
    cb.set_label("면적당 월세(만원/㎡)")
    ax.set_xlabel("임대면적(㎡)")
    ax.set_ylabel("월세(만원)")
    ax.set_title("면적 대비 월세 - 색이 진할수록(값이 낮을수록) 가성비 좋음")
    plt.tight_layout()
    if save:
        plt.savefig(f"{OUTPUT_DIR}/06_rent_vs_area_scatter.png")
    plt.show()


# ========================================================================
# 7. 메인 파이프라인
# ========================================================================
def main():
    raw = load_data(INPUT_PATH)
    df = preprocess(raw, include_semi_jeonse=True)

    # 정제 데이터 저장
    df.to_csv(f"{OUTPUT_DIR}/preprocessed_rent_data.csv", index=False, encoding="utf-8-sig")

    # 자치구 / 법정동 단위 요약
    gu_summary = summarize_by_region(df, "gu", min_n=30)
    dong_summary = summarize_by_region(df, "dong", min_n=30)
    gu_summary.to_csv(f"{OUTPUT_DIR}/summary_by_gu.csv", index=False, encoding="utf-8-sig")
    dong_summary.to_csv(f"{OUTPUT_DIR}/summary_by_dong.csv", index=False, encoding="utf-8-sig")

    # 가성비 스코어
    gu_score = compute_value_score(gu_summary)
    gu_score.to_csv(f"{OUTPUT_DIR}/gu_value_score.csv", index=False, encoding="utf-8-sig")

    # 시각화
    plot_price_distribution(df)
    plot_by_gu(gu_summary)
    plot_cheap_vs_value(gu_score)
    yearly = plot_yearly_trend(df)
    yearly.to_csv(f"{OUTPUT_DIR}/yearly_trend.csv", index=False, encoding="utf-8-sig")
    plot_by_housing_type(df)
    plot_rent_vs_area_scatter(df)

    print("\n[완료] 결과물은 output/ 폴더에 저장되었습니다.")
    print(" - preprocessed_rent_data.csv : 정제된 원자료")
    print(" - summary_by_gu.csv / summary_by_dong.csv : 지역 단위 요약통계")
    print(" - gu_value_score.csv : '싼 동네' vs '가성비 동네' 순위 비교")
    print(" - yearly_trend.csv : 연도별 추이 및 상승률")
    print(" - 01~06 *.png : 시각화 결과")


if __name__ == "__main__":
    main()
