"""目的変数の逆変換、ベースライン、東証カレンダー・SQ日・配当落ち日、OSE CSV。"""
import numpy as np
import pandas as pd
import pytest

from src.baselines import simple_baselines
from src.calendar_jp import TSECalendar, last_us_weekday_before, nyse_holidays, rule_is_open
from src.datasource import OSENightCSVSource
from src.target import make_target, to_price


def _aligned():
    idx = pd.DatetimeIndex(["2026-10-07", "2026-10-08"])
    return pd.DataFrame({"n225_close": [70000.0, 70500.0], "niy_close": [70300.0, 70100.0],
                         "nkd_close": [70250.0, 70150.0], "basis_ma20": [0.001, -0.002]}, index=idx)


@pytest.mark.parametrize("target", ["T1", "T2"])
def test_target_roundtrip(target):
    al = _aligned()
    actual = pd.Series([70683.98, 70035.71], index=al.index)
    y = make_target(actual, al, target)
    base = al["n225_close"] if target == "T1" else al["niy_close"]
    np.testing.assert_allclose(to_price(y, base), actual)


def test_baselines():
    al = _aligned()
    b = simple_baselines(al)
    np.testing.assert_allclose(b["B0"], al["n225_close"])
    np.testing.assert_allclose(b["B1"], al["niy_close"])
    np.testing.assert_allclose(b["B1p"], al["niy_close"] * np.exp(al["basis_ma20"]))


def _cal(cfg, start="2015-01-01", end="2026-10-08"):
    # 過去分はルールで作った営業日を「実績」とみなす
    days = [d for d in pd.date_range(start, end) if rule_is_open(d, cfg)]
    return TSECalendar(days, cfg)


def test_rule_calendar(cfg):
    assert not rule_is_open(pd.Timestamp("2026-12-31"), cfg)
    assert not rule_is_open(pd.Timestamp("2027-01-01"), cfg)
    assert not rule_is_open(pd.Timestamp("2027-01-04") - pd.Timedelta(days=1), cfg)  # 1/3（日）
    assert rule_is_open(pd.Timestamp("2027-01-04"), cfg)
    assert not rule_is_open(pd.Timestamp("2026-09-22"), cfg)  # 国民の休日


def test_history_wins_over_rule(cfg):
    # 2020-10-01 は終日売買停止：実績に無ければ休場扱い
    days = [d for d in pd.date_range("2020-09-01", "2020-10-31") if rule_is_open(d, cfg)
            and d != pd.Timestamp("2020-10-01")]
    cal = TSECalendar(days, cfg)
    assert not cal.is_open("2020-10-01")
    assert cal.is_open("2020-10-02")


def test_sq_dates(cfg):
    cal = _cal(cfg)
    assert cal.sq_date(2024, 3) == pd.Timestamp("2024-03-08")
    assert cal.sq_date(2022, 2) == pd.Timestamp("2022-02-10")   # 2/11（金）は祝日 → 前営業日


def test_exdiv_dates(cfg):
    cal = _cal(cfg)
    assert cal.exdiv_date(2024, 3) == pd.Timestamp("2024-03-28")
    assert cal.exdiv_date(2025, 9) == pd.Timestamp("2025-09-29")
    assert cal.exdiv_date(2019, 3) == pd.Timestamp("2019-03-27")   # 受渡し T+3 時代
    assert cal.exdiv_date(2019, 9) == pd.Timestamp("2019-09-27")


def test_nyse_holidays():
    assert pd.Timestamp("2026-04-03").date() in nyse_holidays(2026)   # Good Friday
    assert pd.Timestamp("2026-11-26").date() in nyse_holidays(2026)   # Thanksgiving
    assert last_us_weekday_before("2026-04-06") == pd.Timestamp("2026-04-02")


def test_ose_csv_source(tmp_path):
    rows = []
    for dt, c, cm, v in [
        ("2026-10-07 16:30", 70000, "2026-12", 10), ("2026-10-08 02:00", 70200, "2026-12", 10),
        ("2026-10-08 05:59", 70300, "2026-12", 10), ("2026-10-08 05:59", 70250, "2027-03", 1),
        ("2026-10-08 10:00", 99999, "2026-12", 10),   # 日中セッションは除外
    ]:
        rows.append({"datetime": dt, "open": c, "high": c, "low": c, "close": c, "volume": v, "contract_month": cm})
    p = tmp_path / "ose.csv"
    pd.DataFrame(rows).to_csv(p, index=False)
    df = OSENightCSVSource(p).fetch()
    assert list(df.index) == [pd.Timestamp("2026-10-08")]
    assert df.loc["2026-10-08", "close"] == 70300
    assert df.loc["2026-10-08", "high"] == 70300
    assert df.loc["2026-10-08", "available_at_utc"] == pd.Timestamp("2026-10-07 21:00", tz="UTC")
