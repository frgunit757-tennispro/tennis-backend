"""
Tennis Analyzer - Финальный сервер
- Рейтинги игроков по Elo
- Прогноз матча (ML модель)
- H2H статистика
- Матчи с коэффициентами букмекеров (the-odds-api)
- 3 критерия: исход, тотал, сеты
- Gemini ИИ анализ с реальной статистикой 2024-2026
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import numpy as np
import pickle
import requests
from pathlib import Path
from contextlib import asynccontextmanager

# ─── Константы ───
DATA         = Path(".")
ODDS_API_KEY = "7f2d9a6e51688c0e68bce9abca2876ba"
GEMINI_KEY   = "AIzaSyCYob0b6nZ5doKQhDS64esHYES6_8m_bMo"
GEMINI_URL   = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={GEMINI_KEY}"

TENNIS_SPORTS = [
    "tennis",
    "tennis_atp_italian_open",
    "tennis_wta_italian_open",
]

# ─── Глобальные данные ───
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


@asynccontextmanager
async def lifespan(app):
    load_data()
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


def get_player_stats(player_name: str) -> dict:
    """Базовая статистика (все годы) — для value-анализа"""
    if atp_df is None:
        return {}
    mask = (
        atp_df["winner_name"].str.contains(player_name, case=False, na=False) |
        atp_df["loser_name"].str.contains(player_name, case=False, na=False)
    )
    matches = atp_df[mask].dropna(subset=["score"]).tail(50)
    total_games = []
    three_set = 0
    wins  = int(atp_df[mask & atp_df["winner_name"].str.contains(player_name, case=False, na=False)].shape[0])
    total = int(atp_df[mask].shape[0])

    for _, row in matches.iterrows():
        score = str(row.get("score", ""))
        games = 0
        set_count = 0
        for s in score.split():
            try:
                p = s.split("-")
                if len(p) == 2:
                    g1 = int(p[0].split("(")[0])
                    g2 = int(p[1].split("(")[0])
                    games += g1 + g2
                    set_count += 1
            except:
                pass
        if games > 0:
            total_games.append(games)
        if set_count >= 3:
            three_set += 1

    return {
        "avg_total":     round(np.mean(total_games), 1) if total_games else None,
        "three_set_pct": round(three_set / len(matches) * 100, 1) if matches.shape[0] > 0 else None,
        "win_rate":      round(wins / total * 100, 1) if total > 0 else None,
        "matches_total": total,
    }


def get_deep_stats_2024(player_name: str, surface: str = "clay") -> dict:
    """
    Глубокая статистика из CSV за 2024-2026.
    Передаётся в Gemini как реальные данные для анализа.
    """
    if atp_df is None:
        return {}

    as_winner = atp_df["winner_name"].str.contains(player_name, case=False, na=False)
    as_loser  = atp_df["loser_name"].str.contains(player_name, case=False, na=False)
    all_m = atp_df[as_winner | as_loser].copy()

    if all_m.empty:
        return {}

    recent = all_m[all_m["tourney_date"].dt.year >= 2024].copy()
    if recent.empty:
        return {}

    wins   = recent[recent["winner_name"].str.contains(player_name, case=False, na=False)]
    losses = recent[recent["loser_name"].str.contains(player_name, case=False, na=False)]
    total  = len(recent)

    # По покрытиям
    surface_stats = {}
    for surf in ["Hard", "Clay", "Grass"]:
        sm = recent[recent["surface"] == surf]
        if len(sm) > 0:
            w = sm[sm["winner_name"].str.contains(player_name, case=False, na=False)]
            surface_stats[surf] = {
                "wins": len(w), "total": len(sm),
                "win_pct": round(len(w) / len(sm) * 100, 1)
            }

    # Тотал и 3 сета
    total_games_list = []
    three_set_count  = 0
    for _, row in recent.iterrows():
        score = str(row.get("score", ""))
        games = 0
        set_count = 0
        for s in score.split():
            try:
                p = s.split("-")
                if len(p) == 2:
                    games += int(p[0].split("(")[0]) + int(p[1].split("(")[0])
                    set_count += 1
            except:
                pass
        if games > 0:
            total_games_list.append(games)
        if set_count >= 3:
            three_set_count += 1

    # Статистика подачи (когда выигрывал)
    srv = {}
    try:
        w_srv = wins[["w_ace","w_df","w_svpt","w_1stIn","w_1stWon","w_bpSaved","w_bpFaced"]].dropna()
        if not w_srv.empty and w_srv["w_svpt"].sum() > 0:
            srv = {
                "aces_per_match":  round(w_srv["w_ace"].mean(), 1),
                "first_serve_pct": round(w_srv["w_1stIn"].sum() / w_srv["w_svpt"].sum() * 100, 1),
                "first_serve_won": round(w_srv["w_1stWon"].sum() / w_srv["w_1stIn"].sum() * 100, 1) if w_srv["w_1stIn"].sum() > 0 else None,
                "bp_saved_pct":    round(w_srv["w_bpSaved"].sum() / w_srv["w_bpFaced"].sum() * 100, 1) if w_srv["w_bpFaced"].sum() > 0 else None,
            }
    except:
        pass

    # Последние 10 матчей
    last10 = []
    for _, row in recent.sort_values("tourney_date").tail(10).iterrows():
        won = player_name.lower() in str(row["winner_name"]).lower()
        opp = str(row["loser_name"] if won else row["winner_name"]).split()[-1]
        last10.append(f"{'W' if won else 'L'} vs {opp} ({row.get('surface','?')[:2]}) {row.get('score','')}")

    # Последние 5 на нужном покрытии
    surf5 = []
    surf_map = {"clay": "Clay", "hard": "Hard", "grass": "Grass"}
    sf = surf_map.get(surface, "Clay")
    for _, row in recent[recent["surface"] == sf].sort_values("tourney_date").tail(5).iterrows():
        won = player_name.lower() in str(row["winner_name"]).lower()
        opp = str(row["loser_name"] if won else row["winner_name"]).split()[-1]
        surf5.append(f"{'W' if won else 'L'} vs {opp} {row.get('score','')}")

    return {
        "total_matches":   total,
        "wins":            len(wins),
        "losses":          len(losses),
        "win_pct":         round(len(wins) / total * 100, 1) if total > 0 else 0,
        "surface_stats":   surface_stats,
        "avg_total_games": round(np.mean(total_games_list), 1) if total_games_list else None,
        "three_set_pct":   round(three_set_count / total * 100, 1) if total > 0 else None,
        "serve":           srv,
        "last_10":         last10,
        "last_5_surface":  surf5,
    }


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


def gemini_analyze(home: str, away: str, our_prob: float,
                   avg_total: float, total_line: float,
                   three_set_pct: float, stats1: dict, stats2: dict,
                   value_h2h: str, value_total: str,
                   deep1: dict = None, deep2: dict = None,
                   surface: str = "clay",
                   odds_home: float = None, odds_away: float = None,
                   total_over: float = None, total_under: float = None) -> str:
    """Gemini анализ: реальная статистика 2024-2026 + ML-модель + коэффициенты"""
    try:
        prob1   = round(our_prob * 100) if our_prob else "?"
        prob2   = 100 - prob1 if isinstance(prob1, int) else "?"
        surf_ru = {"clay": "грунт", "hard": "хард", "grass": "трава"}.get(surface, surface)

        def fmt_deep(name: str, d: dict) -> str:
            if not d:
                return f"  {name}: данные за 2024-2026 не найдены\n"
            surf_map = {"clay": "Clay", "hard": "Hard", "grass": "Grass"}
            sd = d.get("surface_stats", {}).get(surf_map.get(surface, "Clay"), {})
            srv = d.get("serve", {})
            srv_str = (
                f"1я подача {srv.get('first_serve_pct','?')}% | "
                f"выигрыш 1й {srv.get('first_serve_won','?')}% | "
                f"эйсы/матч {srv.get('aces_per_match','?')} | "
                f"БП спасено {srv.get('bp_saved_pct','?')}%"
            ) if srv else "нет данных"
            last10 = " | ".join(d.get("last_10", [])) or "нет"
            surf5  = " | ".join(d.get("last_5_surface", [])) or "нет"
            return (
                f"  {name} (2024-2026, матчей: {d.get('total_matches',0)}):\n"
                f"  • Всего W/L: {d.get('wins','?')}/{d.get('losses','?')} ({d.get('win_pct','?')}%)\n"
                f"  • На {surf_ru}: {sd.get('wins','?')}W из {sd.get('total','?')}M ({sd.get('win_pct','?')}%)\n"
                f"  • Avg тотал: {d.get('avg_total_games','?')} | % в 3 сета: {d.get('three_set_pct','?')}%\n"
                f"  • Подача: {srv_str}\n"
                f"  • Последние 10: {last10}\n"
                f"  • Последние 5 на {surf_ru}: {surf5}\n"
            )

        deep1_txt = fmt_deep(home, deep1 or {})
        deep2_txt = fmt_deep(away, deep2 or {})

        avg_t1 = (deep1 or {}).get("avg_total_games")
        avg_t2 = (deep2 or {}).get("avg_total_games")
        avg_total_best = round((avg_t1 + avg_t2) / 2, 1) if avg_t1 and avg_t2 else avg_total

        three1 = (deep1 or {}).get("three_set_pct")
        three2 = (deep2 or {}).get("three_set_pct")
        three_best = round((three1 + three2) / 2, 1) if three1 and three2 else three_set_pct

        prompt = f"""Ты профессиональный теннисный аналитик. Используй реальную статистику из базы ATP 2024-2026 как основу анализа.

МАТЧ: {home} vs {away} | Покрытие: {surf_ru.upper()}

━━ РЕАЛЬНАЯ СТАТИСТИКА ATP 2024-2026 ━━
{deep1_txt}
{deep2_txt}
━━ ML-МОДЕЛЬ ━━
• Вероятность: {home} {prob1}% | {away} {prob2}%
• Avg тотал (оба): {avg_total_best} | % в 3 сета: {three_best}%

━━ КОЭФФИЦИЕНТЫ БУКМЕКЕРА ━━
• {home}: {odds_home or '—'} | {away}: {odds_away or '—'}
• Тотал {total_line or '—'}: больше {total_over or '—'} / меньше {total_under or '—'}

━━ ПРЕДВАРИТЕЛЬНЫЙ АНАЛИЗ ━━
• Исход: {value_h2h or 'нет данных'}
• Тотал: {value_total or 'нет данных'}

Дай краткий конкретный анализ по 3 критериям на русском. Опирайся на цифры из базы, добавь знания о травмах и форме.

ФОРМАТ (строго, кратко):
🏆 Исход: [кто и почему, 1-2 предложения. Value у букмекера?]
📊 Тотал: [больше/меньше {total_line or '?'} и почему, 1 предложение]
🎯 Сет аутсайдера: [возьмёт ли хотя бы 1 сет, 1 предложение]
✅ Итог: [одна чёткая рекомендация]

⚠️ Не финансовый совет."""

        body = {
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {"temperature": 0.6, "maxOutputTokens": 500}
        }
        r = requests.post(GEMINI_URL, json=body, timeout=20)
        if r.status_code == 200:
            return r.json()["candidates"][0]["content"]["parts"][0]["text"].strip()
        return None
    except Exception as e:
        print(f"Gemini ошибка: {e}")
        return None


# ─── ЭНДПОИНТЫ ───

@app.api_route("/", methods=["GET", "HEAD"])
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

    stats1 = get_player_stats(r1["player"])
    stats2 = get_player_stats(r2["player"])
    avg_total, three_set = None, None
    if stats1.get("avg_total") and stats2.get("avg_total"):
        avg_total = round((stats1["avg_total"] + stats2["avg_total"]) / 2, 1)
    if stats1.get("three_set_pct") and stats2.get("three_set_pct"):
        three_set = round((stats1["three_set_pct"] + stats2["three_set_pct"]) / 2, 1)

    deep1 = get_deep_stats_2024(r1["player"], surface)
    deep2 = get_deep_stats_2024(r2["player"], surface)

    ai_analysis = gemini_analyze(
        r1["player"], r2["player"], prob1,
        avg_total, None, three_set,
        stats1, stats2, None, None,
        deep1=deep1, deep2=deep2, surface=surface
    )

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
        "three_set_pct": three_set,
        "ai_analysis":   ai_analysis,
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
    """Матчи с коэффициентами + 3 критерия + Gemini ИИ анализ с реальной статистикой 2024-2026"""
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
                                if o["name"] == "Over":    total_over  = o["price"]
                                elif o["name"] == "Under": total_under = o["price"]
                        elif market["key"] == "spreads" and not spread_home:
                            for o in market["outcomes"]:
                                if o["name"] == home: spread_home = {"point": o.get("point"), "price": o["price"]}
                                else:                 spread_away = {"point": o.get("point"), "price": o["price"]}
                    break

                our_prob    = None
                avg_total   = None
                three_set_p = None
                value_h2h   = None
                value_total = None
                value_set   = None
                ai_analysis = None
                stats1      = {}
                stats2      = {}

                if ratings_df is not None and model_bundle is not None:
                    r1 = find_player(home.split()[-1])
                    r2 = find_player(away.split()[-1])

                    if r1 is not None and r2 is not None:
                        our_prob = predict_winner(r1, r2, "clay")
                        stats1   = get_player_stats(r1["player"])
                        stats2   = get_player_stats(r2["player"])

                        # Глубокая статистика 2024-2026
                        deep1 = get_deep_stats_2024(r1["player"], "clay")
                        deep2 = get_deep_stats_2024(r2["player"], "clay")

                        # Avg тотал — приоритет данным 2024-2026
                        avg_t1 = deep1.get("avg_total_games")
                        avg_t2 = deep2.get("avg_total_games")
                        if avg_t1 and avg_t2:
                            avg_total = round((avg_t1 + avg_t2) / 2, 1)
                        elif stats1.get("avg_total") and stats2.get("avg_total"):
                            avg_total = round((stats1["avg_total"] + stats2["avg_total"]) / 2, 1)

                        three1 = deep1.get("three_set_pct")
                        three2 = deep2.get("three_set_pct")
                        if three1 and three2:
                            three_set_p = round((three1 + three2) / 2, 1)
                        elif stats1.get("three_set_pct") and stats2.get("three_set_pct"):
                            three_set_p = round((stats1["three_set_pct"] + stats2["three_set_pct"]) / 2, 1)

                        # Value анализ
                        if our_prob and h2h_odds.get(home):
                            v = analyze_value(our_prob, h2h_odds[home])
                            if v["has_value"]:
                                value_h2h = f"✅ {home.split()[-1]} (+{v['diff']}%)"
                            elif h2h_odds.get(away) and analyze_value(1 - our_prob, h2h_odds[away])["has_value"]:
                                v2 = analyze_value(1 - our_prob, h2h_odds[away])
                                value_h2h = f"✅ {away.split()[-1]} (+{v2['diff']}%)"
                            else:
                                value_h2h = "⚪ Нет value"

                        if avg_total and total_line:
                            if avg_total > float(total_line):
                                value_total = f"📊 БОЛЬШЕ {total_line} (avg: {avg_total})"
                            else:
                                value_total = f"📊 МЕНЬШЕ {total_line} (avg: {avg_total})"

                        if three_set_p is not None:
                            value_set = (
                                f"🎾 3 сета ({three_set_p}%)" if three_set_p > 50
                                else f"🎾 2 сета ({100 - three_set_p:.0f}%)"
                            )

                        # Gemini — с реальной статистикой 2024-2026
                        ai_analysis = gemini_analyze(
                            home, away, our_prob,
                            avg_total, total_line, three_set_p,
                            stats1, stats2, value_h2h, value_total,
                            deep1=deep1, deep2=deep2, surface="clay",
                            odds_home=h2h_odds.get(home), odds_away=h2h_odds.get(away),
                            total_over=total_over, total_under=total_under
                        )

                all_matches.append({
                    "home":          home,
                    "away":          away,
                    "date":          m["commence_time"][:10],
                    "time":          m["commence_time"][11:16],
                    "bookmaker":     bookmaker,
                    "tournament":    sport.replace("tennis_", "").replace("_", " ").title(),
                    "odds_home":     h2h_odds.get(home),
                    "odds_away":     h2h_odds.get(away),
                    "total_line":    total_line,
                    "total_over":    total_over,
                    "total_under":   total_under,
                    "spread_home":   spread_home,
                    "spread_away":   spread_away,
                    "our_prob":      round(our_prob, 4) if our_prob else None,
                    "avg_total":     avg_total,
                    "three_set_pct": three_set_p,
                    "value_h2h":     value_h2h,
                    "value_total":   value_total,
                    "value_set":     value_set,
                    "ai_analysis":   ai_analysis,
                })

        except Exception as e:
            print(f"Ошибка {sport}: {e}")
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
    uvicorn.run(app, host="0.0.0.0", port=8000)
