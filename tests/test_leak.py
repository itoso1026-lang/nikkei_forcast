"""リーク検査：確定時刻の assert、米国日付ルール、未来のデータを消しても特徴量が変わらないこと。"""
import numpy as np
import pandas as pd
import pytest

from conftest import make_daily, real_cache_available
from src.align import LeakError, asof_align, assert_no_leak, assign_trade_date, cutoff_utc


def test_cutoff_is_8am_jst(cfg):
    c = cutoff_utc(["2026-10-08"], cfg)[0]
    assert c == pd.Timestamp("2026-10-07 23:00", tz="UTC")


def test_us_bar_of_previous_calendar_day_is_used(cfg):
    # 2026-10-08（木）の予測：米国 10/7 の終値は使え、10/8 の終値は使えない
    spx = make_daily("^GSPC", ["2026-10-06", "2026-10-07", "2026-10-08"], [1.0, 2.0, 3.0], cfg)
    a = asof_align(spx, ["2026-10-08"], "spx", cfg)
    assert a.loc["2026-10-08", "spx_close"] == 2.0
    assert a.loc["2026-10-08", "spx_src_date"] == pd.Timestamp("2026-10-07")


def test_japan_holiday_monday_uses_us_monday(cfg):
    # 2026-09-21（月）〜23 は日本の連休。9/24（木）の予測では、N225 は 9/18、米国は 9/23 を使う
    n225 = make_daily("^N225", ["2026-09-17", "2026-09-18", "2026-09-24"], [1.0, 2.0, 3.0], cfg)
    spx = make_daily("^GSPC", ["2026-09-18", "2026-09-21", "2026-09-22", "2026-09-23"], [1.0, 2.0, 3.0, 4.0], cfg)
    n = asof_align(n225, ["2026-09-24"], "n225", cfg)
    s = asof_align(spx, ["2026-09-24"], "spx", cfg)
    assert n.loc["2026-09-24", "n225_src_date"] == pd.Timestamp("2026-09-18")
    assert s.loc["2026-09-24", "spx_src_date"] == pd.Timestamp("2026-09-23")


def test_jpy_x_uses_t_minus_2(cfg):
    # JPY=X の日足は翌日 00:00 London に確定するので、t の 8:00 JST には t-1 の足はまだ使えない
    for t, expect in (("2026-10-08", "2026-10-06"), ("2026-01-15", "2026-01-13")):
        u = make_daily("JPY=X", pd.bdate_range("2025-12-01", "2026-10-09"), 1.0, cfg)
        a = asof_align(u, [t], "usdjpy", cfg)
        assert a.loc[t, "usdjpy_src_date"] == pd.Timestamp(expect)


def test_cme_bar_dated_t_is_excluded(cfg):
    # CME の bar 日付 t は東証の t 日の取引時間を含むので、t の予測には使えない（冬時間・夏時間とも）
    for t in ("2026-10-08", "2026-01-15"):
        niy = make_daily("NIY=F", pd.bdate_range(pd.Timestamp(t) - pd.Timedelta(days=7), t), 1.0, cfg)
        a = asof_align(niy, [t], "niy", cfg)
        assert a.loc[t, "niy_src_date"] == pd.Timestamp(t) - pd.offsets.BDay(1)


def test_row_available_exactly_at_cutoff_is_excluded(cfg):
    df = pd.DataFrame({"close": [1.0, 2.0]}, index=pd.DatetimeIndex(["2026-10-06", "2026-10-07"]))
    df["available_at_utc"] = [pd.Timestamp("2026-10-06 00:00", tz="UTC"), cutoff_utc(["2026-10-08"], cfg)[0]]
    a = asof_align(df, ["2026-10-08"], "x", cfg)
    assert a.loc["2026-10-08", "x_close"] == 1.0


def test_assert_no_leak_raises(cfg):
    t = pd.DatetimeIndex(["2026-10-08"])
    bad = pd.DataFrame({"x_src_avail": [pd.Timestamp("2026-10-08 00:00", tz="UTC")],
                        "x_src_date": [pd.Timestamp("2026-10-07")]}, index=t)
    with pytest.raises(LeakError):
        assert_no_leak(bad, cfg)
    bad2 = pd.DataFrame({"x_src_avail": [pd.Timestamp("2026-10-07 00:00", tz="UTC")],
                         "x_src_date": [pd.Timestamp("2026-10-08")]}, index=t)
    with pytest.raises(LeakError):
        assert_no_leak(bad2, cfg)
    assert_no_leak(bad2, cfg, same_day_ok=("x",))


def test_assign_trade_date_cme(cfg):
    # CME：日足の区切りは 15:00 CT（清算値の時刻）。それより後に終わる足は翌取引日に属する
    starts = pd.DatetimeIndex(["2026-10-07 19:00", "2026-10-07 20:00", "2026-10-07 23:00"], tz="UTC")
    # 19:00 UTC = 14:00 CDT（終了 15:00 → 10/7）、20:00 UTC = 15:00 CDT（終了 16:00 → 10/8）
    td = assign_trade_date(starts, "NIY=F", cfg=cfg)
    assert list(td.strftime("%Y-%m-%d")) == ["2026-10-07", "2026-10-08", "2026-10-08"]


@pytest.mark.skipif(not real_cache_available(), reason="data/raw のキャッシュがありません")
@pytest.mark.parametrize("t", ["2020-03-17", "2024-08-06", "2026-09-24"])
def test_features_unchanged_when_future_data_removed(cfg, t):
    """t の cutoff 以降に確定したデータを全て消しても、t の特徴量は変わらない（＝未来を見ていない）。"""
    from src.features import build, feature_columns, load_sources

    src = load_sources(cfg)
    t = pd.Timestamp(t)
    cut = cutoff_utc([t], cfg)[0]
    dates = src["n225"].index[(src["n225"].index <= t) & (src["n225"].index >= t - pd.Timedelta(days=400))]
    full = build(dates, cfg, src)
    # N225 の t の行（目的変数）も消える。カレンダー特徴量は t 以降をルール（祝日表）で判定する
    trunc_src = {k: v[v["available_at_utc"] < cut] for k, v in src.items()}
    trunc = build(dates, cfg, trunc_src)
    fc = feature_columns(full)
    a, b = full.loc[t, fc].astype(float), trunc.loc[t, fc].astype(float)
    diff = ~np.isclose(a.values, b.values, equal_nan=True)
    assert not diff.any(), f"未来のデータで変わる特徴量: {list(np.array(fc)[diff])}"
