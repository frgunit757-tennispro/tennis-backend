"""
Tennis Analyzer Backend
- FastAPI + Telegram бот в фоновом потоке
- Gemini AI анализ: наши CSV-данные 2024-2026 + знания Gemini
- 3 критерия: победа / тотал / победа в сете
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import numpy as np
import pickle
import requests
import httpx
import os
import asyncio
import threading
from pathlib import Path
from contextlib import asynccontextmanager
from datetime import datetime

from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup, WebAppInfo
from telegram.ext import Application, CommandHandler, ContextTypes

# ─── Конфиг ───
DATA         = Path(".")
ODDS_API_KEY = os.getenv("ODDS_API_KEY", "7f2d9a6e51688c0e68bce9abca2876ba")
BOT_TOKEN    = os.getenv("TELEGRAM_BOT_TOKEN", "8500214628:AAGInqfRQ9Bsn4cZzTLrysyrXFR9gwvMKNc")
MINI_APP_URL = os.getenv("MINI_APP_URL", "https://frgunit757-tennispro.github.io/tennis-backend/index.html")
GEMINI_KEY   = os.getenv("GEMINI_API_KEY", "AIzaSyCYob0b6nZ5doKQhDS64esHYES6_8m_bMo")
BACKEND_URL  = os.getenv("BACKEND_URL", "https://tennis-bot-tk9q.onrender.com")

GEMINI_URL = (
    "https://generativelanguage.googleapis.com/v1beta/models/"
    f"gemini-2.0-flash:generateContent?key={GEMINI_KEY}"
)

TENNIS_SPORTS = [
    "tennis",
    "tennis_atp_italian_open",
    "tennis_wta_italian_open",
]

ratings_df:   pd.DataFrame = None
atp_df:       pd.DataFrame = None
model_bundle: dict         = None


# ─── Загрузка данных ───

def load_data():
    global ratings_df, atp_df, model_bundle
    try:
        ratings_df = pd.read_csv(DATA / "ratings.csv")
        atp_df     = pd.read_csv(DATA / "atp_all.csv", parse_dates=["tourney_date"])
        with open(DATA / "model.pkl", "rb") as f:
            model_bundle = pickle.load(f)
        print(f"✓ Загружено {len(ratings_df)} игроков, {len(atp_df):,} матчей")
    except Exception as e:
        print(f"Ошибка загрузки данных: {e}")


# ─── Хелперы ───

def find_player(name: str):
    if ratings_df is None:
        return None
    exact = ratings_df[ratings_df["player"] == name]
    if not exact.empty:
        return exact.iloc[0]
    for part in name.split():
        if len(part) > 3:
            partial = ratings_df[ratings_df["player"].str.contains(part, case=False, na=False)]
            if not partial.empty:
                return partial.iloc[0]
    return None


def get_elo(row, surface: str) -> float:
    col_map = {"hard": "elo_hard", "clay": "elo_clay", "grass": "elo_grass"}
    col = col_map.get(surface, "elo_hard")
    return float(row[col]) if col in row.index else float(row["elo_hard"])


def predict_winner(r1, r2, surface: str) -> float:
    if model_bundle is None:
        return None
    elo1 = get_elo(r1, surface)
    elo2 = get_elo(r2, surface)
    elo_prob = 1 / (1 + 10 ** (-(elo1 - elo2) / 400))
    clf, scaler, features = model_bundle["clf"], model_bundle["scaler"], model_bundle["features"]
    X = pd.DataFrame([{
        "elo_diff":  elo1 - elo2,
        "form_diff": float(r1["form"]) - float(r2["form"]),
        "elo_prob":  elo_prob,
        "rank_diff": 0,
        "is_clay":   int(surface == "clay"),
        "is_grass":  int(surface == "grass"),
        "round_num": 4,
    }])[features]
    return float(clf.predict_proba(scaler.transform(X))[0][1])


def analyze_value(our_prob: float, bk_odds: float) -> dict:
    if not our_prob or not bk_odds:
        return {"has_value": False, "diff": 0}
    bk_prob = 1 / bk_odds
    diff = our_prob - bk_prob
    return {
        "has_value":    diff > 0.05,
        "diff":         round(diff * 100, 1),
        "our_prob_pct": round(our_prob * 100, 1),
        "bk_prob_pct":  round(bk_prob * 100, 1),
    }


# ─── ГЛУБОКАЯ СТАТИСТИКА ИЗ CSV ───

def get_deep_stats(player_name: str, surface: str = None) -> dict:
    """
    Вытаскивает полную статистику игрока из CSV за 2024-2026.
    Возвращает словарь с реальными числами для передачи в Gemini.
    """
    if atp_df is None:
        return {}

    # Все матчи игрока
    as_winner = atp_df["winner_name"].str.contains(player_name, case=False, na=False)
    as_loser  = atp_df["loser_name"].str.contains(player_name, case=False, na=False)
    all_matches = atp_df[as_winner | as_loser].copy()

    if all_matches.empty:
        return {}

    # Только 2024-2026
    recent = all_matches[all_matches["tourney_date"].dt.year >= 2024].copy()
    if recent.empty:
        # Если нет данных за 2024+, берём последние 2 года от последней даты
        last_date = all_matches["tourney_date"].max()
        cutoff = last_date - pd.DateOffset(years=2)
        recent = all_matches[all_matches["tourney_date"] >= cutoff].copy()

    if recent.empty:
        return {}

    # ─ Победы/поражения ─
    wins   = recent[recent["winner_name"].str.contains(player_name, case=False, na=False)]
    losses = recent[recent["loser_name"].str.contains(player_name, case=False, na=False)]
    total  = len(recent)
    win_pct = round(len(wins) / total * 100, 1) if total > 0 else 0

    # ─ По покрытиям ─
    surface_stats = {}
    for surf in ["Hard", "Clay", "Grass"]:
        surf_matches = recent[recent["surface"] == surf]
        if len(surf_matches) > 0:
            w = surf_matches[surf_matches["winner_name"].str.contains(player_name, case=False, na=False)]
            surface_stats[surf.lower()] = {
                "wins":    len(w),
                "total":   len(surf_matches),
                "win_pct": round(len(w) / len(surf_matches) * 100, 1)
            }

    # ─ Тотал геймов ─
    total_games_list = []
    three_set_count  = 0
    for _, row in recent.iterrows():
        score = str(row.get("score", ""))
        sets  = score.split()
        games = 0
        set_count = 0
        for s in sets:
            try:
                parts = s.split("-")
                if len(parts) == 2:
                    g1 = int(parts[0].split("(")[0])
                    g2 = int(parts[1].split("(")[0])
                    games += g1 + g2
                    set_count += 1
            except:
                pass
        if games > 0:
            total_games_list.append(games)
        if set_count >= 3:
            three_set_count += 1

    avg_total    = round(np.mean(total_games_list), 1) if total_games_list else None
    three_set_pct = round(three_set_count / total * 100, 1) if total > 0 else None

    # ─ Сервисная статистика (из колонок winner/loser) ─
    ace_col   = "w_ace"   if True else "l_ace"
    df_col    = "w_df"
    svpt_col  = "w_svpt"
    first_col = "w_1stIn"
    won1_col  = "w_1stWon"
    bp_saved  = "w_bpSaved"
    bp_faced  = "w_bpFaced"

    # Считаем сервис когда игрок выигрывал
    wins_srv = wins[[ace_col, df_col, svpt_col, first_col, won1_col, bp_saved, bp_faced]].dropna()
    losses_srv = losses[["l_ace","l_df","l_svpt","l_1stIn","l_1stWon","l_bpSaved","l_bpFaced"]].dropna()
    losses_srv.columns = [ace_col, df_col, svpt_col, first_col, won1_col, bp_saved, bp_faced]

    all_srv = pd.concat([wins_srv, losses_srv], ignore_index=True)

    srv_stats = {}
    if not all_srv.empty and all_srv[svpt_col].sum() > 0:
        total_svpt = all_srv[svpt_col].sum()
        srv_stats = {
            "avg_aces_per_match":    round(all_srv[ace_col].mean(), 1),
            "avg_df_per_match":      round(all_srv[df_col].mean(), 1),
            "first_serve_in_pct":    round(all_srv[first_col].sum() / total_svpt * 100, 1),
            "first_serve_won_pct":   round(all_srv[won1_col].sum() / all_srv[first_col].sum() * 100, 1) if all_srv[first_col].sum() > 0 else None,
            "bp_saved_pct":          round(all_srv[bp_saved].sum() / all_srv[bp_faced].sum() * 100, 1) if all_srv[bp_faced].sum() > 0 else None,
        }

    # ─ Последние 10 матчей (форма) ─
    last10 = recent.sort_values("tourney_date").tail(10)
    last10_results = []
    for _, row in last10.iterrows():
        won = player_name.lower() in str(row["winner_name"]).lower()
        last10_results.append({
            "date":    str(row["tourney_date"].date()),
            "tourney": row.get("tourney_name", ""),
            "surface": row.get("surface", ""),
            "vs":      row["loser_name"] if won else row["winner_name"],
            "result":  "W" if won else "L",
            "score":   row.get("score", ""),
        })

    # ─ Последние 5 матчей на конкретном покрытии ─
    surface_recent = []
    if surface:
        surf_map = {"clay": "Clay", "hard": "Hard", "grass": "Grass"}
        surf_name = surf_map.get(surface, surface.capitalize())
        surf_last = recent[recent["surface"] == surf_name].sort_values("tourney_date").tail(5)
        for _, row in surf_last.iterrows():
            won = player_name.lower() in str(row["winner_name"]).lower()
            surface_recent.append({
                "date":   str(row["tourney_date"].date()),
                "vs":     row["loser_name"] if won else row["winner_name"],
                "result": "W" if won else "L",
                "score":  row.get("score", ""),
            })

    return {
        "matches_total_2024_26":   total,
        "wins":                    len(wins),
        "losses":                  len(losses),
        "win_pct":                 win_pct,
        "surface_stats":           surface_stats,
        "avg_total_games":         avg_total,
        "three_set_pct":           three_set_pct,
        "serve":                   srv_stats,
        "last_10_matches":         last10_results,
        "last_5_on_surface":       surface_recent,
        "data_period":             "2024-2026",
    }


def get_player_total_stats(player_name: str) -> dict:
    """Оставляем для /predict и /upcoming"""
    if atp_df is None:
        return {}
    mask = (
        atp_df["winner_name"].str.contains(player_name, case=False, na=False) |
        atp_df["loser_name"].str.contains(player_name, case=False, na=False)
    )
    player_matches = atp_df[mask].dropna(subset=["score"]).tail(50)
    total_games = []
    three_set_matches = 0
    for _, row in player_matches.iterrows():
        score = str(row.get("score", ""))
        sets = score.split()
        games = 0
        set_count = 0
        for s in sets:
            try:
                parts = s.split("-")
                if len(parts) == 2:
                    g1 = int(parts[0].split("(")[0])
                    g2 = int(parts[1].split("(")[0])
                    games += g1 + g2
                    set_count += 1
            except:
                pass
        if games > 0:
            total_games.append(games)
        if set_count >= 3:
            three_set_matches += 1
    if not total_games:
        return {}
    return {
        "avg_total":      round(np.mean(total_games), 1),
        "three_set_pct":  round(three_set_matches / len(player_matches) * 100, 1),
        "matches_analyzed": len(total_games),
    }


# ─── Форматирование статистики для промпта ───

def format_stats_for_prompt(name: str, s: dict, surface: str) -> str:
    if not s:
        return f"{name}: данных нет\n"

    surf_map = {"clay": "Clay", "hard": "Hard", "grass": "Grass"}
    surf_key = surf_map.get(surface, "Hard").lower()
    surf_data = s.get("surface_stats", {}).get(surf_key, {})

    # Последние 10
    last10 = s.get("last_10_matches", [])
    last10_str = " ".join([f"{m['result']}({m['surface'][:1]})" for m in last10]) if last10 else "нет данных"

    # Последние 5 на покрытии
    surf5 = s.get("last_5_on_surface", [])
    surf5_str = ""
    if surf5:
        surf5_str = "\n".join([
            f"    {m['date']} vs {m['vs']}: {m['result']} {m['score']}"
            for m in surf5
        ])
    else:
        surf5_str = "    нет данных"

    # Сервис
    srv = s.get("serve", {})
    srv_str = "нет данных"
    if srv:
        srv_str = (
            f"1я подача: {srv.get('first_serve_in_pct','?')}% | "
            f"Выигрыш 1й подачи: {srv.get('first_serve_won_pct','?')}% | "
            f"Эйсы/матч: {srv.get('avg_aces_per_match','?')} | "
            f"БП спасено: {srv.get('bp_saved_pct','?')}%"
        )

    return f"""
  {name} (данные 2024-2026, {s.get('matches_total_2024_26',0)} матчей):
  • Общий W/L: {s.get('wins','?')}/{s.get('losses','?')} ({s.get('win_pct','?')}% побед)
  • На {surface}: {surf_data.get('wins','?')}W/{surf_data.get('total','?')}M ({surf_data.get('win_pct','?')}%)
  • Avg тотал геймов: {s.get('avg_total_games','?')} | % матчей в 3 сета: {s.get('three_set_pct','?')}%
  • Сервис: {srv_str}
  • Последние 10 матчей (W/L покрытие): {last10_str}
  • Последние 5 на {surface}:
{surf5_str}"""


# ─── Gemini AI анализ ───

async def gemini_analyze(p1: str, p2: str, surface: str, stats: dict, odds: dict = None,
                          deep1: dict = None, deep2: dict = None) -> str:
    surface_ru  = {"clay": "грунт", "hard": "хард", "grass": "трава"}
    surface_txt = surface_ru.get(surface, surface)

    prob1     = stats.get("prob_p1")
    prob2     = stats.get("prob_p2")
    elo1      = stats.get("elo1", "?")
    elo2      = stats.get("elo2", "?")
    form1     = stats.get("form1", "?")
    form2     = stats.get("form2", "?")
    wins1     = stats.get("wins1", "?")
    wins2     = stats.get("wins2", "?")
    h2h_total = stats.get("total", 0)

    if odds is None:
        odds = {}

    odds_home   = odds.get("odds_home", "—")
    odds_away   = odds.get("odds_away", "—")
    total_line  = odds.get("total_line", "?")
    total_over  = odds.get("total_over", "—")
    total_under = odds.get("total_under", "—")
    spread_home = odds.get("spread_home")
    spread_away = odds.get("spread_away")
    bookmaker   = odds.get("bookmaker", "букмекер")

    spread_text = "нет данных"
    if spread_home and spread_away:
        spread_text = (
            f"{p1}: {spread_home.get('point')} @ {spread_home.get('price')} | "
            f"{p2}: {spread_away.get('point')} @ {spread_away.get('price')}"
        )

    # Форматируем глубокую статистику из CSV
    deep1_txt = format_stats_for_prompt(p1, deep1 or {}, surface)
    deep2_txt = format_stats_for_prompt(p2, deep2 or {}, surface)

    # Средний тотал из глубоких данных
    avg_t1 = (deep1 or {}).get("avg_total_games")
    avg_t2 = (deep2 or {}).get("avg_total_games")
    avg_total_combined = round((avg_t1 + avg_t2) / 2, 1) if avg_t1 and avg_t2 else "нет данных"

    three1 = (deep1 or {}).get("three_set_pct")
    three2 = (deep2 or {}).get("three_set_pct")
    three_combined = round((three1 + three2) / 2, 1) if three1 and three2 else "нет данных"

    prompt = f"""Ты — профессиональный теннисный аналитик и бетинг-эксперт.
Тебе предоставлена РЕАЛЬНАЯ статистика игроков из базы данных ATP (2024-2026),
а также коэффициенты букмекеров. Используй ЭТИ ДАННЫЕ как основу анализа,
дополняя своими актуальными знаниями о форме и новостях.

═══════════════════════════════════════
МАТЧ: {p1} vs {p2}
ПОКРЫТИЕ: {surface_txt.upper()}
═══════════════════════════════════════

━━━ РЕАЛЬНАЯ СТАТИСТИКА ИЗ НАШЕЙ БАЗЫ (2024-2026) ━━━
{deep1_txt}
{deep2_txt}

━━━ НАША ML-МОДЕЛЬ ━━━
• Вероятность победы {p1}: {round(prob1*100,1) if prob1 else "?"}%
• Вероятность победы {p2}: {round(prob2*100,1) if prob2 else "?"}%
• Elo {p1}: {elo1}  |  Elo {p2}: {elo2}
• Форма {p1}: {form1}  |  Форма {p2}: {form2}
• H2H всего: {p1} {wins1}–{wins2} {p2} ({h2h_total} матчей)
• Средний тотал (оба игрока): {avg_total_combined}
• % матчей в 3 сета (среднее): {three_combined}%

━━━ КОЭФФИЦИЕНТЫ БУКМЕКЕРА ({bookmaker}) ━━━
• Победа {p1}: {odds_home}
• Победа {p2}: {odds_away}
• Тотал геймов — линия: {total_line}  (больше: {total_over} / меньше: {total_under})
• Гандикап: {spread_text}

═══════════════════════════════════════
ЗАДАЧА:
1. Опирайся ПРЕЖДЕ ВСЕГО на реальные цифры из базы данных выше
2. Дополни своими знаниями о последних новостях, травмах, форме
3. Проведи анализ по 3 критериям и укажи где есть value у букмекера
═══════════════════════════════════════

ФОРМАТ ОТВЕТА (строго, с эмодзи, на русском):

🎾 *{p1} vs {p2}* | {surface_txt.upper()}

📊 *{p1} (статистика 2024-2026):*
[Опиши реальную форму: W/L на покрытии, последние матчи, сильные/слабые стороны из данных]

📊 *{p2} (статистика 2024-2026):*
[Опиши реальную форму: W/L на покрытии, последние матчи, сильные/слабые стороны из данных]

📰 *Актуальные новости:*
[Добавь из своих знаний — травмы, форма в последних турнирах, мотивация]

━━━━━━━━━━━━━━━━━━━━

🏆 *КРИТЕРИЙ 1 — Победа в матче:*
[Анализ: кто фаворит по данным + ML-модель {round(prob1*100,1) if prob1 else "?"}% vs коэффициент {odds_home}. Есть ли value?]
📌 Рекомендация: [имя @ коэф ✅ / нет value ⚪]

📊 *КРИТЕРИЙ 2 — Тотал геймов:*
[Наш avg тотал = {avg_total_combined}. Линия = {total_line}. Анализ стиля обоих игроков по данным]
📌 Рекомендация: [БОЛЬШЕ {total_line} @ {total_over} ✅ / МЕНЬШЕ @ {total_under} ✅ / нет value ⚪]

🎯 *КРИТЕРИЙ 3 — Победа в сете (аутсайдер):*
[% матчей в 3 сета = {three_combined}%. Анализ на основе реальных данных — берёт ли аутсайдер сет?]
📌 Рекомендация: [ДА @ коэф ✅ / НЕТ / нет данных ⚪]

━━━━━━━━━━━━━━━━━━━━

✅ *ИТОГОВЫЙ ВЕРДИКТ:*
• 🏆 Победитель: ...
• 📊 Тотал: ...
• 🎯 Сет аутсайдера: ...
• 💡 Лучшая ставка: ...

⚠️ _Анализ основан на реальной статистике ATP 2024-2026 + ML-модели + Gemini AI. Не является финансовым советом._"""

    payload = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {
            "temperature":     0.6,
            "maxOutputTokens": 1800,
        }
    }

    async with httpx.AsyncClient(timeout=30) as client:
        resp = await client.post(GEMINI_URL, json=payload)
        resp.raise_for_status()
        data = resp.json()
        return data["candidates"][0]["content"]["parts"][0]["text"]


# ─── Получить данные матча ───

async def fetch_match_data(p1: str, p2: str, surface: str) -> tuple:
    stats = {}
    odds  = {}
    try:
        async with httpx.AsyncClient(timeout=15) as client:
            r_pred = await client.get(f"{BACKEND_URL}/predict", params={"p1": p1, "p2": p2, "surface": surface})
            r_h2h  = await client.get(f"{BACKEND_URL}/h2h",     params={"p1": p1, "p2": p2})
            if r_pred.status_code == 200:
                stats.update(r_pred.json())
            if r_h2h.status_code == 200:
                stats.update(r_h2h.json())
            r_up = await client.get(f"{BACKEND_URL}/upcoming")
            if r_up.status_code == 200:
                for m in r_up.json():
                    h = m.get("home", "").lower()
                    a = m.get("away", "").lower()
                    if p1.lower() in h or p1.lower() in a or p2.lower() in h or p2.lower() in a:
                        odds = m
                        break
    except Exception as e:
        print(f"Ошибка fetch_match_data: {e}")
    return stats, odds


# ─── Telegram Bot хендлеры ───

async def cmd_start(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    keyboard = [[
        InlineKeyboardButton(
            "🎾 Открыть Tennis Analyzer",
            web_app=WebAppInfo(url=MINI_APP_URL)
        )
    ]]
    await update.message.reply_text(
        "👋 Привет!\n\n"
        "🎾 *Tennis Analyzer* — анализ матчей ATP\n\n"
        "• Elo-рейтинги по покрытиям\n"
        "• Прогнозы с вероятностями\n"
        "• H2H история игроков\n"
        "• 🤖 AI-анализ: реальная статистика 2024-2026 + Gemini\n\n"
        "Используй команду:\n"
        "`/analyze Sinner Alcaraz clay`",
        parse_mode="Markdown",
        reply_markup=InlineKeyboardMarkup(keyboard)
    )


async def cmd_help(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "❓ *Как пользоваться:*\n\n"
        "*/analyze Игрок1 Игрок2 покрытие*\n"
        "Примеры:\n"
        "`/analyze Sinner Alcaraz clay`\n"
        "`/analyze Djokovic Zverev hard`\n"
        "`/analyze Swiatek Sabalenka clay`\n\n"
        "Покрытия: `clay` / `hard` / `grass`\n\n"
        "Анализ включает:\n"
        "📊 Реальная статистика 2024-2026 из базы ATP\n"
        "🤖 ML-модель (Elo + форма)\n"
        "💬 Актуальные знания Gemini AI\n"
        "💰 Реальные коэффициенты букмекеров\n\n"
        "3 критерия:\n"
        "🏆 Победа в матче\n"
        "📊 Тотал геймов\n"
        "🎯 Победа аутсайдера в сете\n\n"
        "⚠️ Не финансовый совет.",
        parse_mode="Markdown"
    )


async def cmd_analyze(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    args = ctx.args
    if not args or len(args) < 2:
        await update.message.reply_text(
            "❌ Укажи двух игроков:\n"
            "`/analyze Sinner Alcaraz clay`\n\n"
            "Покрытия: clay / hard / grass",
            parse_mode="Markdown"
        )
        return

    surface = "clay"
    if args[-1].lower() in ("clay", "hard", "grass"):
        surface = args[-1].lower()

    player1 = args[0]
    player2 = args[1]
    surface_ru = {"clay": "грунт 🟤", "hard": "хард 🔵", "grass": "трава 🟢"}

    msg = await update.message.reply_text(
        f"🔍 *Анализирую матч...*\n"
        f"*{player1}* vs *{player2}* | {surface_ru.get(surface)}\n\n"
        f"⏳ Загружаю статистику 2024-2026 + коэффициенты + Gemini AI...",
        parse_mode="Markdown"
    )

    try:
        # Получаем всё параллельно
        stats_task = fetch_match_data(player1, player2, surface)
        stats, odds = await stats_task

        # Глубокая статистика из CSV (синхронно, быстро)
        deep1 = get_deep_stats(player1, surface)
        deep2 = get_deep_stats(player2, surface)

        # Gemini анализ со всеми данными
        analysis = await gemini_analyze(player1, player2, surface, stats, odds, deep1, deep2)

        if len(analysis) > 4000:
            analysis = analysis[:4000] + "\n\n_... (сокращено)_"

        await msg.edit_text(analysis, parse_mode="Markdown")

    except httpx.HTTPStatusError as e:
        await msg.edit_text(
            f"❌ Ошибка Gemini API ({e.response.status_code})\n"
            "Проверьте GEMINI_API_KEY в Render → Environment."
        )
    except Exception as e:
        await msg.edit_text(
            f"❌ Ошибка: {str(e)[:300]}\n\n"
            "Проверьте имена игроков (на английском)."
        )


# ─── Запуск бота в фоновом потоке ───

def run_bot():
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    async def start_polling():
        tg_app = Application.builder().token(BOT_TOKEN).build()
        tg_app.add_handler(CommandHandler("start",   cmd_start))
        tg_app.add_handler(CommandHandler("help",    cmd_help))
        tg_app.add_handler(CommandHandler("analyze", cmd_analyze))
        print("✓ Telegram бот запущен (polling)")
        await tg_app.initialize()
        await tg_app.start()
        await tg_app.updater.start_polling()
        while True:
            await asyncio.sleep(3600)

    loop.run_until_complete(start_polling())


# ─── FastAPI lifespan ───

@asynccontextmanager
async def lifespan(app):
    load_data()
    bot_thread = threading.Thread(target=run_bot, daemon=True)
    bot_thread.start()
    yield


app = FastAPI(title="Tennis Analyzer API", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─── API эндпоинты ───

@app.get("/")
def root():
    return {"status": "ok", "service": "Tennis Analyzer API"}


@app.get("/players")
def get_players():
    if ratings_df is None:
        raise HTTPException(503, "Данные не загружены")
    return sorted(ratings_df[ratings_df["matches_total"] >= 20]["player"].tolist())


@app.get("/ratings")
def get_ratings(surface: str = "hard", limit: int = 30):
    if ratings_df is None:
        raise HTTPException(503, "Данные не загружены")
    col_map = {"hard": "elo_hard", "clay": "elo_clay", "grass": "elo_grass"}
    col = col_map.get(surface, "elo_hard")
    df = ratings_df[ratings_df["matches_total"] >= 30].copy()
    df = df.sort_values(col, ascending=False).head(limit)
    return [
        {
            "player":        row["player"],
            "elo":           round(row[col], 0),
            "form":          round(row["form"], 3),
            "matches_total": int(row["matches_total"]),
        }
        for _, row in df.iterrows()
    ]


@app.get("/predict")
def predict_match(p1: str, p2: str, surface: str = "hard"):
    if ratings_df is None or model_bundle is None:
        raise HTTPException(503, "Данные не загружены")
    r1 = find_player(p1)
    r2 = find_player(p2)
    if r1 is None:
        raise HTTPException(404, f"Игрок '{p1}' не найден")
    if r2 is None:
        raise HTTPException(404, f"Игрок '{p2}' не найден")
    elo1  = get_elo(r1, surface)
    elo2  = get_elo(r2, surface)
    prob1 = predict_winner(r1, r2, surface)
    s1 = get_player_total_stats(r1["player"])
    s2 = get_player_total_stats(r2["player"])
    avg_total = round((s1["avg_total"] + s2["avg_total"]) / 2, 1) if s1.get("avg_total") and s2.get("avg_total") else None
    three_set_pct = round((s1["three_set_pct"] + s2["three_set_pct"]) / 2, 1) if s1.get("three_set_pct") and s2.get("three_set_pct") else None
    return {
        "player1":       r1["player"],
        "player2":       r2["player"],
        "surface":       surface,
        "prob_p1":       round(prob1, 4) if prob1 else None,
        "prob_p2":       round(1 - prob1, 4) if prob1 else None,
        "elo1":          round(elo1, 0),
        "elo2":          round(elo2, 0),
        "form1":         round(float(r1["form"]), 3),
        "form2":         round(float(r2["form"]), 3),
        "avg_total":     avg_total,
        "three_set_pct": three_set_pct,
    }


@app.get("/h2h")
def head_to_head(p1: str, p2: str):
    if atp_df is None:
        raise HTTPException(503, "Данные не загружены")
    r1 = find_player(p1)
    r2 = find_player(p2)
    name1 = r1["player"] if r1 is not None else p1
    name2 = r2["player"] if r2 is not None else p2
    mask = (
        (atp_df["winner_name"].str.contains(p1, case=False, na=False) &
         atp_df["loser_name"].str.contains(p2, case=False, na=False)) |
        (atp_df["winner_name"].str.contains(p2, case=False, na=False) &
         atp_df["loser_name"].str.contains(p1, case=False, na=False))
    )
    h2h = atp_df[mask].sort_values("tourney_date")
    wins1 = int(h2h["winner_name"].str.contains(p1, case=False, na=False).sum())
    wins2 = len(h2h) - wins1
    by_surface = {}
    for surf, grp in h2h.groupby("surface"):
        w1 = int(grp["winner_name"].str.contains(p1, case=False, na=False).sum())
        by_surface[surf] = {"wins1": w1, "wins2": len(grp) - w1, "total": len(grp)}
    recent = []
    for _, row in h2h.tail(8).iterrows():
        recent.append({
            "date":    str(row["tourney_date"].date()),
            "tourney": row.get("tourney_name", ""),
            "surface": row.get("surface", ""),
            "round":   row.get("round", ""),
            "winner":  row["winner_name"],
            "score":   row.get("score", ""),
        })
    return {
        "player1":    name1, "player2":    name2,
        "wins1":      wins1, "wins2":      wins2,
        "total":      len(h2h),
        "by_surface": by_surface,
        "recent":     recent,
    }


@app.get("/upcoming")
def get_upcoming():
    all_matches = []
    for sport in TENNIS_SPORTS:
        try:
            url = (
                f"https://api.the-odds-api.com/v4/sports/{sport}/odds/"
                f"?apiKey={ODDS_API_KEY}&regions=eu&markets=h2h,totals,spreads"
            )
            r = requests.get(url, timeout=10)
            if r.status_code != 200:
                continue
            data = r.json()
            if not isinstance(data, list):
                continue
            for m in data:
                if not m.get("bookmakers"):
                    continue
                home = m["home_team"]
                away = m["away_team"]
                h2h_odds = {}
                total_line = total_over = total_under = None
                spread_home = spread_away = None
                bookmaker = ""
                for bk in m["bookmakers"]:
                    bookmaker = bk["title"]
                    for market in bk["markets"]:
                        if market["key"] == "h2h":
                            for o in market["outcomes"]:
                                h2h_odds[o["name"]] = o["price"]
                        elif market["key"] == "totals" and not total_line:
                            for o in market["outcomes"]:
                                total_line = o.get("point")
                                if o["name"] == "Over":   total_over  = o["price"]
                                elif o["name"] == "Under": total_under = o["price"]
                        elif market["key"] == "spreads" and not spread_home:
                            for o in market["outcomes"]:
                                if o["name"] == home: spread_home = {"point": o.get("point"), "price": o["price"]}
                                else:                 spread_away = {"point": o.get("point"), "price": o["price"]}
                    break
                our_prob = avg_total = three_set_p = None
                value_h2h = value_total = value_set = None
                if ratings_df is not None and model_bundle is not None:
                    r1 = find_player(home.split()[-1])
                    r2 = find_player(away.split()[-1])
                    if r1 is not None and r2 is not None:
                        our_prob = predict_winner(r1, r2, "clay")
                        s1 = get_player_total_stats(r1["player"])
                        s2 = get_player_total_stats(r2["player"])
                        if s1.get("avg_total") and s2.get("avg_total"):
                            avg_total = round((s1["avg_total"] + s2["avg_total"]) / 2, 1)
                        if s1.get("three_set_pct") and s2.get("three_set_pct"):
                            three_set_p = round((s1["three_set_pct"] + s2["three_set_pct"]) / 2, 1)
                        if our_prob and h2h_odds.get(home):
                            v = analyze_value(our_prob, h2h_odds[home])
                            if v["has_value"]:
                                value_h2h = f"✅ {home.split()[-1]} (+{v['diff']}%)"
                            elif analyze_value(1-our_prob, h2h_odds.get(away, 99))["has_value"]:
                                v2 = analyze_value(1-our_prob, h2h_odds.get(away, 99))
                                value_h2h = f"✅ {away.split()[-1]} (+{v2['diff']}%)"
                            else:
                                value_h2h = "⚪ Нет value"
                        if avg_total and total_line:
                            if avg_total > float(total_line):
                                value_total = f"📊 Тотал БОЛЬШЕ {total_line} (avg: {avg_total})"
                            else:
                                value_total = f"📊 Тотал МЕНЬШЕ {total_line} (avg: {avg_total})"
                        if three_set_p is not None:
                            value_set = (f"🎾 Скорее 3 сета ({three_set_p}%)" if three_set_p > 50
                                         else f"🎾 Скорее 2 сета ({100-three_set_p:.0f}%)")
                all_matches.append({
                    "home": home, "away": away,
                    "date": m["commence_time"][:10], "time": m["commence_time"][11:16],
                    "bookmaker": bookmaker,
                    "tournament": sport.replace("tennis_","").replace("_"," ").title(),
                    "odds_home": h2h_odds.get(home), "odds_away": h2h_odds.get(away),
                    "total_line": total_line, "total_over": total_over, "total_under": total_under,
                    "spread_home": spread_home, "spread_away": spread_away,
                    "our_prob": round(our_prob,4) if our_prob else None,
                    "avg_total": avg_total, "three_set_pct": three_set_p,
                    "value_h2h": value_h2h, "value_total": value_total, "value_set": value_set,
                })
        except Exception as e:
            print(f"Ошибка для {sport}: {e}")
    return all_matches


@app.get("/health")
def health():
    return {
        "status":  "ok",
        "players": len(ratings_df) if ratings_df is not None else 0,
        "matches": len(atp_df) if atp_df is not None else 0,
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", 8000)))
