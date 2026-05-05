"""
Бэкенд Tennis Analyzer
- Исход матча
- Тотал геймов (over/under)
- Победа в сете (spreads)
+ Telegram бот запускается в фоновом потоке
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import numpy as np
import pickle
import requests
import os
import asyncio
import threading
from pathlib import Path
from contextlib import asynccontextmanager

# Telegram
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup, WebAppInfo
from telegram.ext import Application, CommandHandler, ContextTypes

DATA         = Path(".")
ODDS_API_KEY = "7f2d9a6e51688c0e68bce9abca2876ba"

BOT_TOKEN    = os.getenv("TELEGRAM_BOT_TOKEN", "8500214628:AAGInqfRQ9Bsn4cZzTLrysyrXFR9gwvMKNc")
MINI_APP_URL = os.getenv("MINI_APP_URL", "https://frgunit757-tennispro.github.io/tennis-backend/index.html")

# Все активные теннисные турниры
TENNIS_SPORTS = [
    "tennis",
    "tennis_atp_italian_open",
    "tennis_wta_italian_open",
]

ratings_df:   pd.DataFrame = None
atp_df:       pd.DataFrame = None
model_bundle: dict         = None


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


# ─── Telegram Bot ───

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
        "• H2H история игроков\n\n"
        "Нажми кнопку чтобы открыть приложение:",
        parse_mode="Markdown",
        reply_markup=InlineKeyboardMarkup(keyboard)
    )


async def cmd_help(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "❓ *Как пользоваться:*\n\n"
        "1. Нажми /start → кнопка 'Открыть'\n"
        "2. Вкладка *Рейтинг* — топ игроков по Elo\n"
        "3. Вкладка *Прогноз* — введи двух игроков и покрытие\n"
        "4. Вкладка *H2H* — история встреч\n\n"
        "⚠️ Прогнозы статистические, не финансовый совет.",
        parse_mode="Markdown"
    )


def run_bot():
    """Запускает Telegram бота в отдельном event loop (фоновый поток)"""
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    async def start_polling():
        app = Application.builder().token(BOT_TOKEN).build()
        app.add_handler(CommandHandler("start", cmd_start))
        app.add_handler(CommandHandler("help",  cmd_help))
        print("✓ Telegram бот запущен (polling)")
        await app.initialize()
        await app.start()
        await app.updater.start_polling()
        # Держим поток живым
        while True:
            await asyncio.sleep(3600)

    loop.run_until_complete(start_polling())


# ─── FastAPI lifespan ───

@asynccontextmanager
async def lifespan(app):
    load_data()
    # Запускаем бота в фоновом потоке
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


# ─── Хелперы ───

def find_player(name: str):
    if ratings_df is None:
        return None
    exact = ratings_df[ratings_df["player"] == name]
    if not exact.empty:
        return exact.iloc[0]
    parts = name.split()
    for part in parts:
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


def get_player_total_stats(player_name: str) -> dict:
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
        "avg_total": round(np.mean(total_games), 1),
        "three_set_pct": round(three_set_matches / len(player_matches) * 100, 1),
        "matches_analyzed": len(total_games),
    }


def analyze_value(our_prob: float, bk_odds: float) -> dict:
    if not our_prob or not bk_odds:
        return {"has_value": False, "diff": 0}
    bk_prob = 1 / bk_odds
    diff = our_prob - bk_prob
    return {
        "has_value": diff > 0.05,
        "diff": round(diff * 100, 1),
        "our_prob_pct": round(our_prob * 100, 1),
        "bk_prob_pct": round(bk_prob * 100, 1),
    }


# ─── ЭНДПОИНТЫ ───

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

    elo1 = get_elo(r1, surface)
    elo2 = get_elo(r2, surface)
    prob1 = predict_winner(r1, r2, surface)

    stats1 = get_player_total_stats(r1["player"])
    stats2 = get_player_total_stats(r2["player"])
    avg_total = None
    if stats1.get("avg_total") and stats2.get("avg_total"):
        avg_total = round((stats1["avg_total"] + stats2["avg_total"]) / 2, 1)

    three_set_pct = None
    if stats1.get("three_set_pct") and stats2.get("three_set_pct"):
        three_set_pct = round((stats1["three_set_pct"] + stats2["three_set_pct"]) / 2, 1)

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
        "player1":    name1,
        "player2":    name2,
        "wins1":      wins1,
        "wins2":      wins2,
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

                h2h_odds    = {}
                total_line  = None
                total_over  = None
                total_under = None
                spread_home = None
                spread_away = None
                bookmaker   = ""

                for bk in m["bookmakers"]:
                    bookmaker = bk["title"]
                    for market in bk["markets"]:
                        if market["key"] == "h2h":
                            for o in market["outcomes"]:
                                h2h_odds[o["name"]] = o["price"]
                        elif market["key"] == "totals" and not total_line:
                            for o in market["outcomes"]:
                                total_line = o.get("point")
                                if o["name"] == "Over":
                                    total_over = o["price"]
                                elif o["name"] == "Under":
                                    total_under = o["price"]
                        elif market["key"] == "spreads" and not spread_home:
                            for o in market["outcomes"]:
                                if o["name"] == home:
                                    spread_home = {"point": o.get("point"), "price": o["price"]}
                                else:
                                    spread_away = {"point": o.get("point"), "price": o["price"]}
                    break

                our_prob    = None
                avg_total   = None
                three_set_p = None
                value_h2h   = None
                value_total = None
                value_set   = None

                if ratings_df is not None and model_bundle is not None:
                    r1 = find_player(home.split()[-1])
                    r2 = find_player(away.split()[-1])

                    if r1 is not None and r2 is not None:
                        our_prob = predict_winner(r1, r2, "clay")

                        stats1 = get_player_total_stats(r1["player"])
                        stats2 = get_player_total_stats(r2["player"])
                        if stats1.get("avg_total") and stats2.get("avg_total"):
                            avg_total = round((stats1["avg_total"] + stats2["avg_total"]) / 2, 1)
                        if stats1.get("three_set_pct") and stats2.get("three_set_pct"):
                            three_set_p = round((stats1["three_set_pct"] + stats2["three_set_pct"]) / 2, 1)

                        if our_prob and h2h_odds.get(home):
                            v = analyze_value(our_prob, h2h_odds[home])
                            if v["has_value"]:
                                value_h2h = f"✅ {home.split()[-1]} (+{v['diff']}%)"
                            elif analyze_value(1 - our_prob, h2h_odds.get(away, 99))["has_value"]:
                                v2 = analyze_value(1 - our_prob, h2h_odds.get(away, 99))
                                value_h2h = f"✅ {away.split()[-1]} (+{v2['diff']}%)"
                            else:
                                value_h2h = "⚪ Нет value"

                        if avg_total and total_line:
                            our_over = 1 if avg_total > float(total_line) else 0
                            if our_over and total_over:
                                value_total = f"📊 Тотал БОЛЬШЕ {total_line} (наш avg: {avg_total})"
                            elif not our_over and total_under:
                                value_total = f"📊 Тотал МЕНЬШЕ {total_line} (наш avg: {avg_total})"

                        if three_set_p is not None:
                            if three_set_p > 50:
                                value_set = f"🎾 Скорее всего 3 сета ({three_set_p}%)"
                            else:
                                value_set = f"🎾 Скорее всего 2 сета ({100-three_set_p:.0f}%)"

                match_data = {
                    "home":         home,
                    "away":         away,
                    "date":         m["commence_time"][:10],
                    "time":         m["commence_time"][11:16],
                    "bookmaker":    bookmaker,
                    "tournament":   sport.replace("tennis_", "").replace("_", " ").title(),
                    "odds_home":    h2h_odds.get(home),
                    "odds_away":    h2h_odds.get(away),
                    "total_line":   total_line,
                    "total_over":   total_over,
                    "total_under":  total_under,
                    "spread_home":  spread_home,
                    "spread_away":  spread_away,
                    "our_prob":     round(our_prob, 4) if our_prob else None,
                    "avg_total":    avg_total,
                    "three_set_pct": three_set_p,
                    "value_h2h":    value_h2h,
                    "value_total":  value_total,
                    "value_set":    value_set,
                }
                all_matches.append(match_data)

        except Exception as e:
            print(f"Ошибка для {sport}: {e}")
            continue

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
