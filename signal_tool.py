#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
signal_tool.py — أداة قرار التداول مع إشعارات شراء/بيع على هاتفك
===================================================================

تفحص كل رمز وتعطيك قرارًا واحدًا: 🟢 شراء / 🔴 بيع / 🟡 احتفاظ / ⚪ انتظار،
ومع الشراء: سعر الدخول، وقف الخسارة، الهدف، والكمية المناسبة لرأس مالك.
وترسل لك إشعارًا على هاتفك (تيليجرام أو ntfy) عند كل إشارة شراء أو بيع.

التثبيت (مرة واحدة):
    pip install yfinance pandas numpy matplotlib

فحص الآن:
    python signal_tool.py AAPL 2222.SR BTC-USD --capital 50000

تقييم السوق وأفضل صفقة (يمسح عشرات الأسهم ويرتبها بدرجة من 100):
    python signal_tool.py --scan saudi --capital 50000        # أو us / crypto / all
    python signal_tool.py --universe-file my_list.txt --capital 50000

مع إشعار تيليجرام:
    python signal_tool.py AAPL 2222.SR BTC-USD --capital 50000 \
        --telegram-token 123456:ABC... --telegram-chat 987654321

مراقبة تلقائية على جهازك (فحص يومي في أوقات تحددها، بتوقيت جهازك):
    python signal_tool.py AAPL 2222.SR BTC-USD --watch --at 15:30,23:59 \
        --telegram-token ... --telegram-chat ...

أدوات الإعداد:
    --get-chat-id TOKEN      يطبع chat id بعد أن ترسل أي رسالة للبوت
    --test-notify            يرسل رسالة تجريبية للتأكد أن الإشعارات تعمل
    --digest                 يرسل ملخص كل الرموز في كل فحص (وليس الشراء/البيع فقط)

القواعد (تتبّع الاتجاه، شراء فقط):
    الاتجاه:   الإغلاق فوق EMA200
    الدخول:    EMA20 تقطع EMA50 لأعلى + RSI(14) بين 40 و70  → التنفيذ عند افتتاح اليوم التالي
    الوقف:     الدخول − 2 × ATR(14)      الهدف: الدخول + 3 × ATR(14)
    الخروج:    الوصول للوقف أو الهدف، أو EMA20 تقطع EMA50 لأسفل
    الحجم:     تخاطر بـ 1% من رأس المال في الصفقة الواحدة (--risk)

تنبيه: قواعد واضحة وإدارة مخاطر ≠ ضمان ربح. الاختبار الرجعي المطبوع مع كل
إشارة هو دليلك الوحيد على جدوى القواعد على تلك الأداة تحديدًا.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

try:  # حتى تظهر العربية والرموز على ويندوز
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:  # noqa: BLE001
    pass


# ───────────────────────────── الإعدادات ─────────────────────────────
@dataclass
class Params:
    fast: int = 20
    slow: int = 50
    trend: int = 200
    rsi_len: int = 14
    rsi_lo: float = 40.0
    rsi_hi: float = 70.0
    atr_len: int = 14
    atr_stop: float = 2.0
    atr_target: float = 3.0
    risk_pct: float = 1.0


ICON = {"شراء": "🟢", "بيع": "🔴", "احتفاظ": "🟡", "انتظار": "⚪"}


# ───────────────────────────── المؤشرات ─────────────────────────────
def ema(series: pd.Series, n: int) -> pd.Series:
    return series.ewm(span=n, adjust=False).mean()


def rsi(close: pd.Series, n: int) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    avg_gain = gain.ewm(alpha=1.0 / n, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1.0 / n, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    out = 100.0 - 100.0 / (1.0 + rs)
    out = out.where(avg_loss != 0.0, 100.0)
    return out.fillna(50.0)


def atr(df: pd.DataFrame, n: int) -> pd.Series:
    prev_close = df["Close"].shift()
    tr = pd.concat(
        [df["High"] - df["Low"], (df["High"] - prev_close).abs(), (df["Low"] - prev_close).abs()],
        axis=1,
    ).max(axis=1)
    return tr.ewm(alpha=1.0 / n, adjust=False).mean()


def add_indicators(df: pd.DataFrame, p: Params) -> pd.DataFrame:
    df = df.copy()
    df["ema_fast"] = ema(df["Close"], p.fast)
    df["ema_slow"] = ema(df["Close"], p.slow)
    df["ema_trend"] = ema(df["Close"], p.trend)
    df["rsi"] = rsi(df["Close"], p.rsi_len)
    df["atr"] = atr(df, p.atr_len)

    above = df["ema_fast"] > df["ema_slow"]
    prev_above = above.shift(fill_value=False).astype(bool)
    cross_up = above & ~prev_above
    cross_dn = ~above & prev_above

    df["buy_signal"] = cross_up & (df["Close"] > df["ema_trend"]) & df["rsi"].between(p.rsi_lo, p.rsi_hi)
    df["sell_signal"] = cross_dn

    warm = max(p.trend, p.slow, p.atr_len)  # فترة إحماء المؤشرات
    df.loc[df.index[:warm], ["buy_signal", "sell_signal"]] = False
    return df


# ───────────────────────────── إدارة المخاطر ─────────────────────────────
def position_size(cash: float, entry: float, stop: float, risk_pct: float, fractional: bool) -> float:
    per_unit_risk = entry - stop
    if per_unit_risk <= 0 or entry <= 0 or cash <= 0:
        return 0.0
    qty = (cash * risk_pct / 100.0) / per_unit_risk
    qty = min(qty, cash / entry)
    return round(qty, 6) if fractional else float(math.floor(qty))


# ───────────────────────────── الاختبار الرجعي ─────────────────────────────
def backtest(df: pd.DataFrame, p: Params, capital: float, fractional: bool):
    cash = capital
    pos = None
    trades = []
    equity = []
    pending = False

    for i in range(len(df)):
        row = df.iloc[i]
        date = df.index[i]

        if pending and pos is None:  # تنفيذ الدخول عند الافتتاح
            entry = float(row["Open"])
            a = float(df["atr"].iloc[i - 1])
            stop = entry - p.atr_stop * a
            target = entry + p.atr_target * a
            qty = position_size(cash, entry, stop, p.risk_pct, fractional)
            if qty > 0:
                pos = {"entry": entry, "stop": stop, "target": target, "qty": qty, "date": date}
                cash -= qty * entry
            pending = False

        if pos is not None:  # فحص الخروج
            exit_price, reason = None, None
            if float(row["Low"]) <= pos["stop"]:
                exit_price, reason = min(pos["stop"], float(row["Open"])), "وقف الخسارة"
            elif float(row["High"]) >= pos["target"]:
                exit_price, reason = max(pos["target"], float(row["Open"])), "تحقق الهدف"
            elif bool(row["sell_signal"]):
                exit_price, reason = float(row["Close"]), "إشارة خروج"
            if exit_price is not None:
                cash += pos["qty"] * exit_price
                trades.append({
                    "entry_date": pos["date"], "exit_date": date,
                    "entry": pos["entry"], "exit": exit_price,
                    "pct": (exit_price / pos["entry"] - 1) * 100,
                    "pnl": (exit_price - pos["entry"]) * pos["qty"],
                    "reason": reason,
                })
                pos = None

        if pos is None and bool(row["buy_signal"]) and not np.isnan(row["atr"]):
            pending = True

        equity.append(cash + (pos["qty"] * float(row["Close"]) if pos else 0.0))

    return trades, pd.Series(equity, index=df.index), pos, pending


def performance(trades, equity: pd.Series, capital: float, df: pd.DataFrame) -> dict:
    wins = [t for t in trades if t["pnl"] > 0]
    losses = [t for t in trades if t["pnl"] <= 0]
    gross_win = sum(t["pnl"] for t in wins)
    gross_loss = -sum(t["pnl"] for t in losses)
    pf = gross_win / gross_loss if gross_loss > 0 else (float("inf") if gross_win > 0 else 0.0)
    return {
        "trades": len(trades),
        "win_rate": len(wins) / len(trades) * 100 if trades else 0.0,
        "avg_win": float(np.mean([t["pct"] for t in wins])) if wins else 0.0,
        "avg_loss": float(np.mean([t["pct"] for t in losses])) if losses else 0.0,
        "profit_factor": pf,
        "total_return": (equity.iloc[-1] / capital - 1) * 100,
        "buy_hold": (float(df["Close"].iloc[-1]) / float(df["Close"].iloc[0]) - 1) * 100,
        "max_dd": float(((equity / equity.cummax()) - 1).min() * 100),
    }


def verdict(perf: dict) -> str:
    if perf["trades"] < 8:
        return "⚠ عدد الصفقات قليل — لا تكفي للحكم على جدوى الإشارات هنا"
    if perf["profit_factor"] >= 1.3 and perf["max_dd"] > -25 and perf["total_return"] > 0:
        return "✔ القواعد أدّت جيدًا تاريخيًا على هذه الأداة — الإشارة تستحق الاعتبار"
    return "✘ أداء تاريخي ضعيف — لا تعتمد على إشارات هذه الأداة بهذه القواعد"


# ───────────────────────────── قرار اليوم + أحداث الإشعار ─────────────────────────────
def fmt(x: float) -> str:
    x = float(x)
    if abs(x) >= 1000:
        return f"{x:,.2f}"
    if abs(x) >= 1:
        return f"{x:.2f}"
    return f"{x:.4f}"


def size_line(qty: float, entry: float, stop: float, p: Params) -> str:
    if qty <= 0:
        return "   الكمية: رأس المال لا يكفي لصفقة بهذا الوقف"
    return (f"   الكمية: {qty:g}   (الخسارة إن ضُرب الوقف ≈ {fmt(qty * (entry - stop))}"
            f" = {p.risk_pct:g}% من رأس المال)")


def decide(df, trades, pos, pending, p: Params, capital: float, fractional: bool):
    """يعيد (القرار، التفصيل، السياق، قائمة الأحداث). كل حدث = (قرار، تفصيل، مفتاح فريد للإشعار)."""
    last = df.iloc[-1]
    d1 = df.index[-1]
    d0 = df.index[-2]
    close = float(last["Close"])
    a = float(last["atr"])
    trend_up = close > float(last["ema_trend"])
    ctx = (f"الإغلاق {fmt(close)} | EMA{p.fast} {fmt(last['ema_fast'])} | EMA{p.slow} {fmt(last['ema_slow'])} | "
           f"EMA{p.trend} {fmt(last['ema_trend'])} | RSI {float(last['rsi']):.0f} | "
           f"الاتجاه العام {'صاعد ↑' if trend_up else 'هابط ↓'}")
    events = []

    # خروج حدث اليوم أو في الجلسة الماضية
    if trades and trades[-1]["exit_date"] in (d0, d1):
        t = trades[-1]
        when = "اليوم" if t["exit_date"] == d1 else "في الجلسة الماضية"
        detail = (f"بِع — أُغلقت الصفقة {when} ({t['reason']}): "
                  f"دخول {fmt(t['entry'])} ← خروج {fmt(t['exit'])} ({t['pct']:+.1f}%)")
        events.append(("بيع", detail, f"sell|{t['exit_date'].date()}"))

    # إشارة شراء ظهرت اليوم (التنفيذ عند الافتتاح القادم)
    if pending:
        stop = close - p.atr_stop * a
        target = close + p.atr_target * a
        qty = position_size(capital, close, stop, p.risk_pct, fractional)
        detail = (f"ادخل عند افتتاح الجلسة القادمة (≈ {fmt(close)})\n"
                  f"   وقف الخسارة: {fmt(stop)}   الهدف: {fmt(target)}   (ربح:مخاطرة {p.atr_target / p.atr_stop:.1f}:1)\n"
                  f"{size_line(qty, close, stop, p)}")
        events.append(("شراء", detail, f"buy|{d1.date()}"))
    # الإشارة ظهرت في الجلسة الماضية ونُفّذ الدخول اليوم عند الافتتاح
    elif pos is not None and pos["date"] == d1:
        qty = position_size(capital, pos["entry"], pos["stop"], p.risk_pct, fractional)
        detail = (f"إشارة الجلسة الماضية — الدخول عند افتتاح اليوم بسعر {fmt(pos['entry'])}"
                  f" (الآن {fmt(close)}، {(close / pos['entry'] - 1) * 100:+.1f}%)\n"
                  f"   وقف الخسارة: {fmt(pos['stop'])}   الهدف: {fmt(pos['target'])}\n"
                  f"{size_line(qty, pos['entry'], pos['stop'], p)}")
        events.append(("شراء", detail, f"buy|{d0.date()}"))

    if events:
        decision, detail, _ = events[-1]
        return decision, detail, ctx, events

    if pos is not None:
        unreal = (close / pos["entry"] - 1) * 100
        detail = (f"صفقة مفتوحة منذ {pos['date'].date()} بسعر {fmt(pos['entry'])} ({unreal:+.1f}% حاليًا)\n"
                  f"   أبقِ وقف الخسارة عند {fmt(pos['stop'])} والهدف عند {fmt(pos['target'])} — لا تتدخل يدويًا")
        return "احتفاظ", detail, ctx, events

    if not trend_up:
        detail = f"السعر تحت EMA{p.trend} (اتجاه هابط) — لا شراء مهما بدا السعر رخيصًا"
    elif float(last["ema_fast"]) > float(last["ema_slow"]):
        detail = "الاتجاه صاعد لكن التقاطع حدث سابقًا — انتظر إشارة دخول جديدة ولا تلاحق السعر"
    else:
        detail = "الاتجاه العام صاعد لكن الزخم القريب ضعيف — انتظر تقاطع EMA لأعلى"
    return "انتظار", detail, ctx, events


# ───────────────────────────── البيانات ─────────────────────────────
def is_crypto(name: str) -> bool:
    return name.upper().endswith(("-USD", "-USDT", "-USDC"))


def load_data(name: str, period: str, csv: str | None) -> pd.DataFrame:
    if csv:
        raw = pd.read_csv(csv)
        raw.columns = [str(c).strip().title() for c in raw.columns]
        date_col = next((c for c in raw.columns if c.lower() in ("date", "datetime", "timestamp", "time")), raw.columns[0])
        raw[date_col] = pd.to_datetime(raw[date_col])
        df = raw.set_index(date_col).sort_index()
    else:
        try:
            import yfinance as yf
        except ImportError:
            sys.exit("المكتبة غير مثبتة. شغّل أولًا:  pip install yfinance")
        df = yf.Ticker(name).history(period=period, auto_adjust=True)
        if df is None or df.empty:
            raise ValueError("لا توجد بيانات لهذا الرمز — تأكد من كتابته كما في Yahoo Finance (مثل 2222.SR أو BTC-USD)")
        df.index = pd.to_datetime(df.index).tz_localize(None)

    need = ["Open", "High", "Low", "Close"]
    missing = [c for c in need if c not in df.columns]
    if missing:
        raise ValueError(f"أعمدة ناقصة في البيانات: {', '.join(missing)}")
    cols = need + (["Volume"] if "Volume" in df.columns else [])
    df = df[cols].astype(float).dropna(subset=need)
    df.index.name = "Date"
    return df


# ───────────────────────────── الإشعارات ─────────────────────────────
def _post_json(url: str, payload: dict) -> bytes:
    req = urllib.request.Request(url, data=json.dumps(payload).encode("utf-8"),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=20) as r:
        return r.read()


def _post_text(url: str, text: str) -> bytes:
    req = urllib.request.Request(url, data=text.encode("utf-8"),
                                 headers={"Content-Type": "text/plain; charset=utf-8"})
    with urllib.request.urlopen(req, timeout=20) as r:
        return r.read()


class Notifier:
    def __init__(self, tg_token=None, tg_chat=None, ntfy_topic=None):
        self.tg_token = tg_token or os.environ.get("TELEGRAM_TOKEN")
        self.tg_chat = tg_chat or os.environ.get("TELEGRAM_CHAT_ID")
        self.ntfy = ntfy_topic or os.environ.get("NTFY_TOPIC")

    @property
    def configured(self) -> bool:
        return bool(self.tg_token and self.tg_chat) or bool(self.ntfy)

    def send(self, text: str) -> bool:
        ok = False
        if self.tg_token and self.tg_chat:
            try:
                _post_json(f"https://api.telegram.org/bot{self.tg_token}/sendMessage",
                           {"chat_id": self.tg_chat, "text": text})
                print("  ✉ أُرسل إشعار تيليجرام")
                ok = True
            except Exception as e:  # noqa: BLE001
                print(f"  ✗ فشل إرسال تيليجرام: {e}")
        if self.ntfy:
            try:
                _post_text(f"https://ntfy.sh/{self.ntfy}", text)
                print("  ✉ أُرسل إشعار ntfy")
                ok = True
            except Exception as e:  # noqa: BLE001
                print(f"  ✗ فشل إرسال ntfy: {e}")
        return ok


def telegram_get_chat_id(token: str):
    try:
        with urllib.request.urlopen(f"https://api.telegram.org/bot{token}/getUpdates", timeout=20) as r:
            data = json.loads(r.read())
    except Exception as e:  # noqa: BLE001
        sys.exit(f"تعذر الاتصال بتيليجرام: {e}")
    chats = {}
    for u in data.get("result", []):
        msg = u.get("message") or u.get("channel_post") or {}
        chat = msg.get("chat")
        if chat:
            chats[chat["id"]] = chat.get("first_name") or chat.get("title") or chat.get("username") or ""
    if not chats:
        print("لم تصل للبوت أي رسالة بعد. افتح البوت في تيليجرام، اضغط Start، أرسل أي كلمة، ثم أعد هذا الأمر.")
    for cid, who in chats.items():
        print(f"chat id: {cid}   ({who})")


def format_alert(name: str, decision: str, detail: str, perf: dict) -> str:
    return (f"{ICON[decision]} {decision} — {name}\n"
            f"{detail}\n"
            f"— اختبار رجعي: فوز {perf['win_rate']:.0f}% | عائد {perf['total_return']:+.0f}% | "
            f"تراجع {perf['max_dd']:.0f}%\n{verdict(perf)}")


def format_digest(results: list) -> str:
    lines = [f"📋 ملخص الفحص {datetime.now():%Y-%m-%d %H:%M}"]
    for r in results:
        lines.append(f"{ICON[r['decision']]} {r['decision']}  {r['name']}  —  إغلاق {fmt(r['df']['Close'].iloc[-1])}")
    return "\n".join(lines)


# ───────────────────────────── حالة الإشعارات (منع التكرار) ─────────────────────────────
def load_state(path: str) -> dict:
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except Exception:  # noqa: BLE001
        return {}


def save_state(path: str, state: dict):
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(state, f, ensure_ascii=False, indent=1)
    except Exception as e:  # noqa: BLE001
        print(f"  (تعذر حفظ ملف الحالة: {e})")


def already_sent(state: dict, name: str, key: str) -> bool:
    return key in state.get(name, [])


def mark_sent(state: dict, name: str, key: str):
    lst = state.setdefault(name, [])
    lst.append(key)
    del lst[:-30]


# ───────────────────────────── العرض ─────────────────────────────
def print_report(name, df, decision, detail, ctx, perf, trades):
    line = "═" * 66
    print(f"\n{line}\n  {name}    (آخر بيانات: {df.index[-1].date()})\n{line}")
    print(f"  القرار اليوم:  {ICON[decision]}  {decision}")
    print(f"  {detail}")
    print(f"\n  {ctx}")
    pf = "∞" if perf["profit_factor"] == float("inf") else f"{perf['profit_factor']:.2f}"
    print(f"\n  الاختبار الرجعي  {df.index[0].date()} → {df.index[-1].date()}  ({len(df)} يوم تداول)")
    print(f"    الصفقات: {perf['trades']}    نسبة الفوز: {perf['win_rate']:.0f}%    عامل الربح: {pf}")
    print(f"    متوسط الصفقة الرابحة: {perf['avg_win']:+.1f}%    متوسط الخاسرة: {perf['avg_loss']:+.1f}%")
    print(f"    عائد القواعد: {perf['total_return']:+.1f}%    الشراء والاحتفاظ: {perf['buy_hold']:+.1f}%"
          f"    أقصى تراجع: {perf['max_dd']:.1f}%")
    print(f"    الحكم: {verdict(perf)}")
    if trades:
        print("\n  آخر الصفقات:")
        for t in trades[-5:]:
            print(f"    {t['entry_date'].date()} → {t['exit_date'].date()}   {fmt(t['entry'])} → {fmt(t['exit'])}"
                  f"   {t['pct']:+.1f}%   ({t['reason']})")


def plot(name, df, trades, equity, p: Params, path: str):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True, gridspec_kw={"height_ratios": [3, 1]})
    ax1.plot(df.index, df["Close"], color="black", lw=1, label="Close")
    ax1.plot(df.index, df["ema_fast"], lw=0.9, label=f"EMA{p.fast}")
    ax1.plot(df.index, df["ema_slow"], lw=0.9, label=f"EMA{p.slow}")
    ax1.plot(df.index, df["ema_trend"], lw=0.9, ls="--", label=f"EMA{p.trend}")
    for i, t in enumerate(trades):
        ax1.scatter(t["entry_date"], t["entry"], marker="^", color="green", s=70, zorder=5, label="Buy" if i == 0 else None)
        ax1.scatter(t["exit_date"], t["exit"], marker="v", color="red", s=70, zorder=5, label="Sell" if i == 0 else None)
    ax1.set_title(f"{name} — signals")
    ax1.legend(loc="upper left")
    ax1.grid(alpha=0.3)
    ax2.plot(equity.index, equity, color="tab:blue")
    ax2.set_title("Strategy equity")
    ax2.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


# ───────────────────────────── مسح السوق: تقييم السوق وأفضل صفقة ─────────────────────────────
UNIVERSES = {
    "us": [
        "AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META", "TSLA", "BRK-B", "AVGO", "JPM", "V", "MA", "UNH",
        "XOM", "LLY", "JNJ", "PG", "HD", "COST", "ABBV", "MRK", "KO", "PEP", "CVX", "WMT", "BAC", "CRM",
        "AMD", "NFLX", "ADBE", "ORCL", "CSCO", "ACN", "TMO", "MCD", "ABT", "LIN", "DHR", "INTU", "TXN",
        "QCOM", "AMGN", "CAT", "GE", "IBM", "HON", "LOW", "UNP", "PFE", "NKE", "BA", "SPGI", "GS", "MS",
        "BLK", "AXP", "DE", "BKNG", "ISRG", "ADP", "MDT", "GILD", "VRTX", "LMT", "RTX", "SBUX", "PLD",
        "SCHW", "C", "TJX", "MMC", "CB", "AMAT", "LRCX", "MU", "PANW", "NOW", "UBER", "PYPL", "SHOP",
        "ABNB", "COIN", "PLTR", "ARM", "MRVL", "KLAC", "ANET", "CRWD", "SNOW", "DIS",
    ],
    "saudi": [
        "2222.SR", "1120.SR", "2010.SR", "7010.SR", "1180.SR", "1010.SR", "1050.SR", "1060.SR", "1080.SR",
        "1150.SR", "1140.SR", "1030.SR", "1111.SR", "1211.SR", "2082.SR", "5110.SR", "7020.SR", "7030.SR",
        "2280.SR", "2050.SR", "2290.SR", "2350.SR", "2310.SR", "2380.SR", "2330.SR", "2060.SR", "2020.SR",
        "4030.SR", "4190.SR", "4003.SR", "4002.SR", "4004.SR", "4013.SR", "4001.SR", "4164.SR", "4163.SR",
        "4161.SR", "4210.SR", "4220.SR", "4250.SR", "4300.SR", "4321.SR", "4240.SR", "4200.SR", "4050.SR",
        "4031.SR", "4261.SR", "4263.SR", "4262.SR", "4280.SR", "4090.SR", "4100.SR", "2081.SR", "2083.SR",
        "2223.SR", "7200.SR", "7202.SR", "7203.SR", "8010.SR", "8210.SR", "8230.SR", "6001.SR", "6010.SR",
        "6002.SR", "6004.SR", "6015.SR", "1810.SR", "1830.SR", "1831.SR", "3030.SR", "3020.SR", "3040.SR",
        "3050.SR", "3060.SR", "3080.SR", "3010.SR", "1201.SR", "1202.SR", "1210.SR", "1212.SR", "1301.SR",
        "1302.SR", "1303.SR", "1304.SR", "2230.SR", "2240.SR", "2250.SR", "2381.SR", "2382.SR", "4142.SR",
        "4144.SR", "4008.SR", "4011.SR", "4015.SR", "4016.SR", "4017.SR", "4020.SR", "4150.SR", "4292.SR",
        "4071.SR", "4072.SR", "4270.SR", "4330.SR", "4340.SR", "1183.SR", "1182.SR", "2001.SR", "2170.SR",
    ],
    "crypto": [
        "BTC-USD", "ETH-USD", "BNB-USD", "SOL-USD", "XRP-USD", "ADA-USD", "DOGE-USD", "AVAX-USD", "DOT-USD",
        "LINK-USD", "LTC-USD", "TRX-USD", "ATOM-USD", "UNI-USD", "XLM-USD", "NEAR-USD", "ETC-USD", "BCH-USD",
        "FIL-USD", "APT-USD", "ARB-USD", "OP-USD", "SUI-USD", "ICP-USD", "HBAR-USD", "AAVE-USD", "INJ-USD",
        "ALGO-USD", "VET-USD", "MKR-USD",
    ],
}
UNIVERSE_LABEL = {"us": "السوق الأمريكية", "saudi": "السوق السعودية (تداول)", "crypto": "العملات الرقمية"}
INDEX_SYMBOL = {"us": "^GSPC", "saudi": "^TASI.SR", "crypto": "BTC-USD"}
MIN_DOLLAR_VOLUME = {"us": 5e6, "saudi": 3e6, "crypto": 0.0}


def load_universe(key: str, custom_file: str | None) -> list:
    if custom_file:
        with open(custom_file, encoding="utf-8") as f:
            return [ln.strip().upper() for ln in f if ln.strip() and not ln.strip().startswith("#")]
    return UNIVERSES[key]


def fetch_universe(tickers: list, period: str) -> dict:
    """تنزيل دفعة واحدة لكل الرموز (أسرع بكثير من رمز رمز)."""
    try:
        import yfinance as yf
    except ImportError:
        sys.exit("المكتبة غير مثبتة. شغّل أولًا:  pip install yfinance")
    data = yf.download(tickers, period=period, interval="1d", auto_adjust=True,
                       group_by="ticker", threads=True, progress=False)
    out = {}
    for t in tickers:
        try:
            df = data[t] if isinstance(data.columns, pd.MultiIndex) else data
        except KeyError:
            continue
        df = df.dropna(subset=["Close"]) if "Close" in df.columns else pd.DataFrame()
        if df.empty:
            continue
        cols = [c for c in ["Open", "High", "Low", "Close", "Volume"] if c in df.columns]
        df = df[cols].astype(float)
        df.index = pd.to_datetime(df.index)
        if getattr(df.index, "tz", None) is not None:
            df.index = df.index.tz_localize(None)
        df.index.name = "Date"
        out[t] = df
    return out


def market_regime(symbol: str, period: str, p: Params):
    """حالة السوق العامة من مؤشره: صاعد / متذبذب / هابط."""
    try:
        df = fetch_universe([symbol], period).get(symbol)
    except Exception:  # noqa: BLE001
        df = None
    if df is None or len(df) < p.trend + 30:
        return None
    dfi = add_indicators(df, p)
    last = dfi.iloc[-1]
    close = float(last["Close"])
    above200 = close > float(last["ema_trend"])
    fast_over_slow = float(last["ema_fast"]) > float(last["ema_slow"])
    label = "صاعد" if (above200 and fast_over_slow) else "هابط" if (not above200 and not fast_over_slow) else "متذبذب"
    ret20 = (close / float(dfi["Close"].iloc[-21]) - 1) * 100
    return {"symbol": symbol, "label": label, "above200": above200, "ret20": ret20}


def evaluate_candidate(name, dfi, trades, pos, pending, perf, p: Params, max_age: int) -> dict:
    last = dfi.iloc[-1]
    close = float(last["Close"])
    a = float(last["atr"])
    ef, es, et = float(last["ema_fast"]), float(last["ema_slow"]), float(last["ema_trend"])
    trend_up = close > et
    slope_up = et > float(dfi["ema_trend"].iloc[-21])
    dist_atr = (close - et) / a if a > 0 else 0.0
    mom63 = close / float(dfi["Close"].iloc[-64]) - 1 if len(dfi) > 64 else 0.0
    atr_pct = a / close * 100 if close > 0 else 99.0
    dvol = None
    if "Volume" in dfi.columns and dfi["Volume"].notna().any():
        dvol = float((dfi["Close"] * dfi["Volume"]).tail(20).mean())

    status, age, signal_date = "none", None, None
    entry = stop = target = None
    if pending:
        status, age, signal_date = "fresh", 0, dfi.index[-1]
        entry, stop, target = close, close - p.atr_stop * a, close + p.atr_target * a
    elif pos is not None:
        sig_idx = dfi.index.get_loc(pos["date"]) - 1
        age = len(dfi) - 1 - sig_idx
        signal_date = dfi.index[sig_idx]
        entry, stop, target = pos["entry"], pos["stop"], pos["target"]
        extended = close > entry + 1.0 * a
        status = "fresh" if (age <= max_age and not extended) else "open"
    elif (trend_up and ef < es and (es - ef) < 0.5 * a
          and ef > float(dfi["ema_fast"].iloc[-4]) and p.rsi_lo <= float(last["rsi"]) <= p.rsi_hi):
        status = "watch"

    def bars(t):
        return dfi.index.get_loc(t["exit_date"]) - dfi.index.get_loc(t["entry_date"])
    win_bars = [bars(t) for t in trades if t["pnl"] > 0]
    all_bars = [bars(t) for t in trades]

    return {
        "name": name, "close": close, "atr": a, "status": status, "age": age, "signal_date": signal_date,
        "entry": entry, "stop": stop, "target": target, "trend_up": trend_up, "slope_up": slope_up,
        "dist_atr": dist_atr, "mom63": mom63, "atr_pct": atr_pct, "dvol": dvol, "perf": perf,
        "gap_atr": (es - ef) / a if a > 0 else 9.0,
        "win_bars": float(np.median(win_bars)) if win_bars else None,
        "all_bars": float(np.median(all_bars)) if all_bars else None,
    }


def score_candidates(rows: list) -> None:
    """درجة من 100 لكل مرشح: ملاءمة القواعد للسهم، الاتجاه، الزخم النسبي، التذبذب، حداثة الإشارة."""
    if not rows:
        return
    mom_rank = pd.Series([r["mom63"] for r in rows]).rank(pct=True).tolist()
    vol_rank = pd.Series([r["atr_pct"] for r in rows]).rank(pct=True).tolist()
    for r, mr, vr in zip(rows, mom_rank, vol_rank):
        perf = r["perf"]
        pf = min(perf["profit_factor"], 3.0) / 3.0
        fit = pf * (30 if perf["trades"] >= 5 else 15)
        if perf["trades"] >= 5 and perf["total_return"] <= 0:
            fit *= 0.5
        trend = 20 if (r["slope_up"] and 0.5 <= r["dist_atr"] <= 6) else 12 if r["slope_up"] else 6
        mom = 20 * mr
        vol = 15 * (1 - vr)
        fresh = {0: 15, 1: 12, 2: 9, 3: 6}.get(r["age"], 3) if r["age"] is not None else 0
        if r["entry"] and r["close"] > r["entry"] + 0.5 * r["atr"]:
            fresh -= 5
        r["score"] = max(0.0, min(100.0, fit + trend + mom + vol + fresh))
        r["mom_pct"] = mr * 100
        r["reasons"] = [
            ("إشارة جديدة اليوم" if r["age"] == 0 else f"إشارة قبل {r['age']} جلسة"),
            f"زخم 3 أشهر {r['mom63'] * 100:+.0f}% (أفضل من {mr * 100:.0f}% من السوق)",
            (f"القواعد ربحت هنا: فوز {perf['win_rate']:.0f}%، عامل ربح {min(perf['profit_factor'], 99):.1f} ({perf['trades']} صفقة)"
             if perf["trades"] >= 5 else f"سجل قصير على هذا السهم ({perf['trades']} صفقة)"),
        ]


def regime_advice(regime, breadth: float, n: int = 100):
    """تقييم البيئة العامة → (نص التقييم، مضاعف الحجم المقترح)."""
    label = regime["label"] if regime else None
    if regime is None and n < 10:
        return "ℹ قائمة صغيرة لا تكفي لتقييم السوق العام — استخدم --scan لتقييم السوق كاملًا", 1.0
    if label == "هابط" or breadth < 35:
        return "✘ بيئة غير مناسبة للشراء — أفضل قرار اليوم: انتظار (الإشارات أدناه للاطلاع فقط)", 0.0
    if label == "صاعد" and breadth >= 50:
        return "✔ بيئة مناسبة للشراء — يمكن الدخول بالحجم الكامل", 1.0
    return "⚠ بيئة متذبذبة — اكتفِ بأفضل صفقة واحدة وبنصف الحجم", 0.5


def sell_plan(r: dict) -> str:
    when = f" — تاريخيًا يتحقق خلال ~{r['win_bars']:.0f} يوم تداول" if r["win_bars"] else ""
    return (f"وقت البيع: (1) عند الهدف {fmt(r['target'])}{when}   (2) عند وقف الخسارة {fmt(r['stop'])} فورًا\n"
            f"              (3) عند إغلاق EMA20 تحت EMA50 — أضف الرمز لقائمة المتابعة لتصلك إشارة البيع تلقائيًا")


def format_scan_alert(ukey: str, regime, breadth: float, advice: str, best: dict, capital: float, mult: float) -> str:
    qty = position_size(capital * mult, best["entry"], best["stop"], 1.0, is_crypto(best["name"])) if mult > 0 else 0
    head = f"🏆 أفضل صفقة — {UNIVERSE_LABEL.get(ukey, ukey)}\n"
    head += f"حالة السوق: {regime['label'] if regime else 'غير معروفة'} | {breadth:.0f}% من الأسهم صاعدة\n{advice}\n\n"
    body = (f"{best['name']}  (درجة {best['score']:.0f}/100)\n"
            f"دخول ≈ {fmt(best['entry'])} | وقف {fmt(best['stop'])} | هدف {fmt(best['target'])}"
            + (f" | الكمية {qty:g}" if qty > 0 else "") + "\n"
            + (f"البيع: عند الهدف (~{best['win_bars']:.0f} يوم) أو الوقف أو تقاطع EMA لأسفل\n" if best["win_bars"]
               else "البيع: عند الهدف أو الوقف أو تقاطع EMA لأسفل\n")
            + "لماذا: " + " | ".join(best["reasons"]))
    return head + body


def _cand_export(r: dict) -> dict:
    perf = r["perf"]
    return {
        "name": r["name"], "score": round(r.get("score", 0.0), 1), "entry": r["entry"], "stop": r["stop"],
        "target": r["target"], "close": r["close"], "age": r["age"],
        "signal_date": str(r["signal_date"].date()) if r["signal_date"] is not None else None,
        "win_bars": r["win_bars"], "reasons": r.get("reasons", []), "fractional": is_crypto(r["name"]),
        "perf": {"win_rate": perf["win_rate"], "total_return": perf["total_return"], "max_dd": perf["max_dd"],
                 "profit_factor": min(perf["profit_factor"], 99.0), "trades": perf["trades"]},
        "verdict": verdict(perf),
    }


def scan_market(args, p: Params, notifier: Notifier, state: dict) -> list:
    keys = ["us", "saudi", "crypto"] if args.scan == "all" else [args.scan]
    summaries = []
    for ukey in keys:
        tickers = load_universe(ukey, args.universe_file)
        label = UNIVERSE_LABEL.get(ukey, ukey) if not args.universe_file else f"قائمتك ({len(tickers)} رمزًا)"
        line = "═" * 66
        print(f"\n{line}\n  🔎 مسح السوق: {label}\n{line}")
        print(f"  جارٍ تنزيل بيانات {len(tickers)} رمزًا...")

        regime = market_regime(INDEX_SYMBOL.get(ukey, ""), args.period, p) if not args.universe_file else None
        try:
            frames = fetch_universe(tickers, args.period)
        except Exception as e:  # noqa: BLE001
            print(f"  ✗ تعذر تنزيل البيانات: {e}")
            continue

        rows = []
        for name, df in frames.items():
            if len(df) < p.trend + 30:
                continue
            try:
                dfi = add_indicators(df, p)
                fractional = is_crypto(name)
                trades, equity, pos, pending = backtest(dfi, p, args.capital, fractional)
                perf = performance(trades, equity, args.capital, dfi)
                rows.append(evaluate_candidate(name, dfi, trades, pos, pending, perf, p, args.max_age))
            except Exception as e:  # noqa: BLE001
                print(f"  (تخطي {name}: {e})")
        if not rows:
            print("  ✗ لا توجد بيانات كافية لأي رمز")
            continue

        min_dvol = MIN_DOLLAR_VOLUME.get(ukey, 0.0)
        for r in rows:
            r["liquid"] = r["dvol"] is None or r["dvol"] >= min_dvol
        breadth = 100.0 * sum(1 for r in rows if r["trend_up"]) / len(rows)
        advice, mult = regime_advice(regime, breadth, len(rows))
        score_candidates(rows)

        if regime:
            print(f"\n  حالة السوق: {'📈' if regime['label'] == 'صاعد' else '📉' if regime['label'] == 'هابط' else '↔'} {regime['label']}"
                  f" — المؤشر {regime['symbol']} {'فوق' if regime['above200'] else 'تحت'} EMA200، تغيّر 20 يومًا {regime['ret20']:+.1f}%")
        print(f"  اتساع السوق: {breadth:.0f}% من {len(rows)} رمزًا في اتجاه صاعد")
        print(f"  التقييم: {advice}")

        fresh = sorted([r for r in rows if r["status"] == "fresh" and r["liquid"]], key=lambda r: -r["score"])
        watch = sorted([r for r in rows if r["status"] == "watch" and r["liquid"]], key=lambda r: r["gap_atr"])
        summaries.append({
            "key": ukey, "label": label, "regime": regime, "breadth": round(breadth, 1), "n": len(rows),
            "advice": advice, "mult": mult,
            "best": _cand_export(fresh[0]) if fresh else None,
            "alternatives": [_cand_export(r) for r in fresh[1:args.top]],
            "watch": [{"name": r["name"], "close": r["close"], "gap_atr": round(r["gap_atr"], 2),
                       "mom63": round(r["mom63"] * 100, 1)} for r in watch[:args.top]],
        })

        if fresh:
            best = fresh[0]
            qty = position_size(args.capital * mult, best["entry"], best["stop"], p.risk_pct, is_crypto(best["name"])) if mult > 0 else 0
            print(f"\n  🏆 أفضل صفقة: {best['name']}   (درجة {best['score']:.0f}/100)")
            print(f"     دخول ≈ {fmt(best['entry'])}   وقف الخسارة {fmt(best['stop'])}   الهدف {fmt(best['target'])}"
                  + (f"   الكمية {qty:g} (مخاطرة {p.risk_pct:g}%{'، نصف الحجم' if mult == 0.5 else ''})" if qty > 0
                     else "   (لا يُنصح بالدخول اليوم)" if mult == 0 else ""))
            print(f"     {sell_plan(best)}")
            print(f"     لماذا: " + " | ".join(best["reasons"]))
            if len(fresh) > 1:
                print("\n  بدائل مرتبة:")
                for i, r in enumerate(fresh[1:args.top], start=2):
                    print(f"     {i}) {r['name']:<10} درجة {r['score']:>3.0f}   دخول {fmt(r['entry'])}   وقف {fmt(r['stop'])}"
                          f"   هدف {fmt(r['target'])}   {r['reasons'][0]}")
            key = f"scan|{ukey}|{best['name']}|{best['signal_date'].date()}"
            if notifier.configured and not already_sent(state, "_scan", key):
                if notifier.send(format_scan_alert(ukey, regime, breadth, advice, best, args.capital, mult)):
                    mark_sent(state, "_scan", key)
        else:
            print("\n  لا توجد إشارة دخول جديدة اليوم في هذا السوق — أفضل قرار: انتظار.")
            if watch:
                print("  أقرب الأسهم لإشارة شراء (راقبها):")
                for r in watch[:args.top]:
                    print(f"     • {r['name']:<10} إغلاق {fmt(r['close'])}   الفجوة حتى التقاطع {r['gap_atr']:.2f} ATR"
                          f"   زخم 3 أشهر {r['mom63'] * 100:+.0f}%")

    save_state(args.state, state)
    return summaries


def write_json(path: str, results: list, scans: list, p: Params):
    """يحفظ نتائج الفحص في ملف JSON يقرؤه تطبيق الهاتف (docs/index.html)."""
    tracked = []
    for r in results:
        perf = r["perf"]
        tracked.append({
            "name": r["name"], "decision": r["decision"], "detail": r["detail"],
            "close": float(r["df"]["Close"].iloc[-1]), "date": str(r["df"].index[-1].date()),
            "verdict": verdict(perf),
            "perf": {"win_rate": perf["win_rate"], "total_return": perf["total_return"], "max_dd": perf["max_dd"],
                     "profit_factor": min(perf["profit_factor"], 99.0), "trades": perf["trades"]},
        })
    payload = {
        "updated": datetime.now().astimezone().isoformat(timespec="minutes"),
        "risk_pct": p.risk_pct, "tracked": tracked, "scans": scans,
    }
    try:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=1, default=str)
        print(f"  💾 حُفظت النتائج للتطبيق: {path}")
    except Exception as e:  # noqa: BLE001
        print(f"  (تعذر حفظ {path}: {e})")


# ───────────────────────────── دورة فحص واحدة ─────────────────────────────
def run_once(args, p: Params, notifier: Notifier, state: dict) -> list:
    items = [(args.name, args.csv)] if args.csv else [(t.upper(), None) for t in args.tickers]
    results = []

    for name, csv in items:
        try:
            df = load_data(name, args.period, csv)
        except Exception as e:  # noqa: BLE001
            print(f"\n✗ {name}: {e}")
            continue
        if len(df) < p.trend + 30:
            print(f"\n✗ {name}: بيانات غير كافية ({len(df)} يوم) — يلزم {p.trend + 30} يومًا على الأقل، جرّب --period 3y")
            continue

        fractional = is_crypto(name)
        df = add_indicators(df, p)
        trades, equity, pos, pending = backtest(df, p, args.capital, fractional)
        perf = performance(trades, equity, args.capital, df)
        decision, detail, ctx, events = decide(df, trades, pos, pending, p, args.capital, fractional)
        print_report(name, df, decision, detail, ctx, perf, trades)

        if args.plot:
            path = f"{name.replace('.', '_').replace('-', '_')}_signals.png"
            try:
                plot(name, df, trades, equity, p, path)
                print(f"\n  📈 الرسم البياني: {path}")
            except Exception as e:  # noqa: BLE001
                print(f"\n  (تعذر إنشاء الرسم: {e})")

        # الإشعارات: شراء/بيع فقط، ومرة واحدة لكل حدث
        for ev_decision, ev_detail, key in events:
            if already_sent(state, name, key):
                continue
            if not notifier.configured:
                print("  💡 لتصلك هذه الإشارة إشعارًا على هاتفك، اضبط تيليجرام أو ntfy (انظر README)")
                break
            if notifier.send(format_alert(name, ev_decision, ev_detail, perf)):
                mark_sent(state, name, key)

        results.append({"name": name, "decision": decision, "detail": detail, "perf": perf, "df": df})

    if len(results) > 1:
        print("\n" + "═" * 66 + "\n  الملخص\n" + "═" * 66)
        for r in results:
            print(f"  {ICON[r['decision']]} {r['decision']:<7} {r['name']:<12} فوز {r['perf']['win_rate']:>3.0f}%  "
                  f"عائد {r['perf']['total_return']:+6.1f}%  تراجع {r['perf']['max_dd']:6.1f}%")

    if args.digest and results and notifier.configured:
        notifier.send(format_digest(results))

    save_state(args.state, state)
    scans = scan_market(args, p, notifier, state) if getattr(args, "scan", None) else []
    if getattr(args, "json", None):
        write_json(args.json, results, scans, p)
    print("\n  تذكير: قرارات مبنية على قواعد وإدارة مخاطر، وليست ضمانًا للربح. التزم بالوقف دائمًا.\n")
    return results


# ───────────────────────────── وضع المراقبة ─────────────────────────────
def parse_times(s: str):
    out = set()
    for part in s.split(","):
        h, m = part.strip().split(":")
        out.add((int(h), int(m)))
    return sorted(out)


def next_run(now: datetime, times) -> datetime:
    for h, m in times:
        cand = now.replace(hour=h, minute=m, second=0, microsecond=0)
        if cand > now:
            return cand
    h, m = times[0]
    return (now + timedelta(days=1)).replace(hour=h, minute=m, second=0, microsecond=0)


def watch(args, p: Params, notifier: Notifier, state: dict):
    times = parse_times(args.at)
    pretty = ", ".join(f"{h:02d}:{m:02d}" for h, m in times)
    print(f"\n👀 وضع المراقبة: فحص يومي عند {pretty} (بتوقيت هذا الجهاز). اضغط Ctrl+C للإيقاف.")
    if not notifier.configured:
        print("   ⚠ لم يُضبط أي إشعار — ستُطبع النتائج هنا فقط.")
    try:
        run_once(args, p, notifier, state)
        while True:
            nxt = next_run(datetime.now(), times)
            print(f"  ⏰ الفحص القادم: {nxt:%Y-%m-%d %H:%M}")
            time.sleep(max(1.0, (nxt - datetime.now()).total_seconds()))
            try:
                run_once(args, p, notifier, state)
            except Exception as e:  # noqa: BLE001
                print(f"  ✗ خطأ أثناء الفحص: {e}")
    except KeyboardInterrupt:
        print("\n  تم إيقاف المراقبة.")


# ───────────────────────────── التشغيل ─────────────────────────────
def main():
    ap = argparse.ArgumentParser(description="أداة قرار التداول مع إشعارات شراء/بيع")
    ap.add_argument("tickers", nargs="*", help="رموز مثل: AAPL 2222.SR BTC-USD")
    ap.add_argument("--csv", help="ملف بيانات بدل الإنترنت (Date,Open,High,Low,Close)")
    ap.add_argument("--name", default="CSV", help="اسم الأداة عند استخدام --csv")
    ap.add_argument("--capital", type=float, default=10000, help="رأس المال (افتراضي 10000)")
    ap.add_argument("--risk", type=float, default=1.0, help="نسبة المخاطرة لكل صفقة %% (افتراضي 1)")
    ap.add_argument("--period", default="3y", help="فترة البيانات: 1y, 2y, 3y, 5y, max")
    ap.add_argument("--fast", type=int, default=20)
    ap.add_argument("--slow", type=int, default=50)
    ap.add_argument("--trend", type=int, default=200)
    ap.add_argument("--plot", action="store_true", help="حفظ رسم بياني PNG لكل رمز")
    # الإشعارات
    ap.add_argument("--telegram-token", help="توكن بوت تيليجرام (أو TELEGRAM_TOKEN)")
    ap.add_argument("--telegram-chat", help="chat id في تيليجرام (أو TELEGRAM_CHAT_ID)")
    ap.add_argument("--ntfy", help="اسم موضوع ntfy.sh سري (أو NTFY_TOPIC)")
    ap.add_argument("--get-chat-id", metavar="TOKEN", help="طباعة chat id بعد إرسال رسالة للبوت")
    ap.add_argument("--test-notify", action="store_true", help="إرسال رسالة تجريبية")
    ap.add_argument("--digest", action="store_true", help="إرسال ملخص كل الرموز في كل فحص")
    ap.add_argument("--state", default="alerts_state.json", help="ملف يمنع تكرار الإشعارات")
    # مسح السوق
    ap.add_argument("--scan", choices=["us", "saudi", "crypto", "all"], help="تقييم السوق وترتيب أفضل الصفقات")
    ap.add_argument("--universe-file", help="ملف نصي برموزك الخاصة (رمز في كل سطر) بدل القائمة المدمجة")
    ap.add_argument("--top", type=int, default=5, help="عدد الصفقات في الترتيب (افتراضي 5)")
    ap.add_argument("--max-age", type=int, default=3, help="أقصى عمر للإشارة بالجلسات لتُعد جديدة (افتراضي 3)")
    ap.add_argument("--json", help="حفظ النتائج بصيغة JSON لتطبيق الهاتف، مثل: --json docs/data.json")
    # المراقبة
    ap.add_argument("--watch", action="store_true", help="مراقبة مستمرة مع فحص يومي")
    ap.add_argument("--at", default="23:59", help="أوقات الفحص اليومي مثل 15:30,23:59")
    args = ap.parse_args()

    if args.get_chat_id:
        telegram_get_chat_id(args.get_chat_id)
        return

    notifier = Notifier(args.telegram_token, args.telegram_chat, args.ntfy)

    if args.test_notify:
        if not notifier.configured:
            sys.exit("اضبط --telegram-token و --telegram-chat (أو --ntfy) أولًا.")
        ok = notifier.send("✅ إشعارات أداة التداول تعمل. ستصلك هنا إشارات الشراء والبيع.")
        sys.exit(0 if ok else 1)

    if not args.tickers and not args.csv and not args.scan and not args.universe_file:
        ap.error("اكتب رمزًا واحدًا على الأقل أو استخدم --scan، مثال:  python signal_tool.py --scan saudi")
    if args.universe_file and not args.scan:
        args.scan = "custom"

    p = Params(fast=args.fast, slow=args.slow, trend=args.trend, risk_pct=args.risk)
    state = load_state(args.state)

    if args.watch:
        watch(args, p, notifier, state)
    else:
        run_once(args, p, notifier, state)


if __name__ == "__main__":
    main()
