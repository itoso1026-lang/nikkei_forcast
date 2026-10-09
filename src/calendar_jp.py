"""東証カレンダー、SQ日、配当落ち日、NYSE 休場日。

過去：^N225 に取引がある日を営業日とする（実績が正）。
将来（実績の最終日より後）：土日・祝日（jpholiday）・12/31・1/2・1/3 を休場とする。
"""
from __future__ import annotations

from datetime import date, timedelta
from functools import lru_cache

import jpholiday
import pandas as pd

from .config import load_config


def rule_is_open(d: date, cfg: dict | None = None) -> bool:
    cfg = cfg or load_config()
    d = pd.Timestamp(d).date()
    if d.weekday() >= 5 or jpholiday.is_holiday(d):
        return False
    return d.strftime("%m-%d") not in cfg["calendar"]["extra_tse_holidays_md"]


class TSECalendar:
    def __init__(self, history_dates, cfg: dict | None = None):
        self.cfg = cfg or load_config()
        h = pd.DatetimeIndex(pd.to_datetime(history_dates)).normalize().unique().sort_values()
        self.history = h
        self.first_hist = h.min() if len(h) else pd.Timestamp("2100-01-01")
        self.last_hist = h.max() if len(h) else pd.Timestamp("1900-01-01")
        self._hist_set = set(h)

    def is_open(self, d) -> bool:
        d = pd.Timestamp(d).normalize()
        if self.first_hist <= d <= self.last_hist:
            return d in self._hist_set
        return rule_is_open(d, self.cfg)

    def days(self, start, end) -> pd.DatetimeIndex:
        """[start, end] の営業日。"""
        start, end = pd.Timestamp(start).normalize(), pd.Timestamp(end).normalize()
        hist = self.history[(self.history >= start) & (self.history <= end)]
        past_end = min(end, self.first_hist - pd.Timedelta(days=1))
        past = [d for d in pd.date_range(start, past_end) if rule_is_open(d, self.cfg)] if start <= past_end else []
        fut_start = max(start, self.last_hist + pd.Timedelta(days=1))
        fut = [d for d in pd.date_range(fut_start, end) if rule_is_open(d, self.cfg)] if fut_start <= end else []
        return pd.DatetimeIndex(past).append(hist).append(pd.DatetimeIndex(fut))

    def prev_open(self, d, n: int = 1) -> pd.Timestamp:
        d = pd.Timestamp(d).normalize()
        while n > 0:
            d -= pd.Timedelta(days=1)
            if self.is_open(d):
                n -= 1
        return d

    def next_open(self, d, n: int = 1) -> pd.Timestamp:
        d = pd.Timestamp(d).normalize()
        while n > 0:
            d += pd.Timedelta(days=1)
            if self.is_open(d):
                n -= 1
        return d

    def sq_date(self, year: int, month: int) -> pd.Timestamp:
        """第2金曜日。休場ならその前営業日。"""
        first = pd.Timestamp(year=year, month=month, day=1)
        first_fri = first + pd.Timedelta(days=(4 - first.weekday()) % 7)
        d = first_fri + pd.Timedelta(days=7)
        return d if self.is_open(d) else self.prev_open(d)

    def month_last_open(self, year: int, month: int) -> pd.Timestamp:
        d = (pd.Timestamp(year=year, month=month, day=1) + pd.offsets.MonthEnd(0)).normalize()
        return d if self.is_open(d) else self.prev_open(d)

    def exdiv_date(self, year: int, month: int) -> pd.Timestamp:
        """配当落ち日 = 権利付最終日の翌営業日。

        権利付最終日は権利確定日（月末最終営業日）の 2営業日前（2019-07-16 以前の受渡しルールでは 3営業日前）。
        """
        record = self.month_last_open(year, month)
        lag = 2 if record >= pd.Timestamp("2019-07-18") else 3
        last_cum = self.prev_open(record, lag)
        return self.next_open(last_cum)


@lru_cache(maxsize=64)
def nyse_holidays(year: int) -> frozenset:
    """NYSE の休場日（主要な祝日のみ。臨時休場は含まない）。"""
    def observed(d: date) -> date:
        if d.weekday() == 5:
            return d - timedelta(days=1)
        if d.weekday() == 6:
            return d + timedelta(days=1)
        return d

    def nth_weekday(month, weekday, n):
        d = date(year, month, 1)
        d += timedelta(days=(weekday - d.weekday()) % 7)
        return d + timedelta(weeks=n - 1)

    def last_weekday(month, weekday):
        d = date(year, month + 1, 1) - timedelta(days=1) if month < 12 else date(year, 12, 31)
        return d - timedelta(days=(d.weekday() - weekday) % 7)

    # イースター（グレゴリオ暦、Anonymous Gregorian algorithm）
    a, b, c = year % 19, year // 100, year % 100
    d_, e = b // 4, b % 4
    f = (b + 8) // 25
    g = (b - f + 1) // 3
    h = (19 * a + b - d_ - g + 15) % 30
    i, k = c // 4, c % 4
    l_ = (32 + 2 * e + 2 * i - h - k) % 7
    m = (a + 11 * h + 22 * l_) // 451
    month = (h + l_ - 7 * m + 114) // 31
    day = ((h + l_ - 7 * m + 114) % 31) + 1
    good_friday = date(year, month, day) - timedelta(days=2)

    hs = {
        observed(date(year, 1, 1)),
        nth_weekday(1, 0, 3),            # MLK
        nth_weekday(2, 0, 3),            # Presidents
        good_friday,
        last_weekday(5, 0),              # Memorial
        observed(date(year, 7, 4)),
        nth_weekday(9, 0, 1),            # Labor
        nth_weekday(11, 3, 4),           # Thanksgiving
        observed(date(year, 12, 25)),
    }
    if year >= 2022:
        hs.add(observed(date(year, 6, 19)))  # Juneteenth
    return frozenset(hs)


def last_us_weekday_before(t) -> pd.Timestamp:
    """暦日で t より前の、直近の NYSE 営業日（祝日ルールによる推定）。"""
    d = pd.Timestamp(t).normalize() - pd.Timedelta(days=1)
    while d.weekday() >= 5 or d.date() in nyse_holidays(d.year):
        d -= pd.Timedelta(days=1)
    return d
