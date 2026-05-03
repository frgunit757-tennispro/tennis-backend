"""
Бэкенд для Telegram Mini App
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
    partial = ratings_df[ratings_df["player"].str.contains(name, case=False, na=False)]
    if not partial.empty:
        return partial.iloc[0]
    return None


def get_elo(row, surface: str) -> float:
    col_map = {"hard": "elo_hard", "clay": "elo_clay", "grass": "elo_grass"}
    col = col_map.get(surface, "elo_hard")
    return float(row[col]) if col in row.index else float(row["elo_hard"])


# ─── ЭНДПОИНТЫ ───

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
    elo_prob = 1 / (1 + 10 ** (-(elo1 - elo2) / 400))

    clf, scaler, features = model_bundle["clf"], model_bundle["scaler"], model_bundle["features"]
    X = pd.DataFrame([{
        "elo_diff":  elo1 - elo2,
        "form_diff": float(r1["form"]) - float(r2["form"]),
        "elo_prob":  elo_prob,
        "rank_diff": 0,
        "is_clay":   int(surface == "clay"),
        "is_grass":  int(surface == "grass"),
        "round_num": 5,
    }])[features]

    prob1 = float(clf.predict_proba(scaler.transform(X))[0][1])

    return {
        "player1": r1["player"],
        "player2": r2["player"],
        "surface": surface,
        "prob_p1": round(prob1, 4),
        "prob_p2": round(1 - prob1, 4),
        "elo1":    round(elo1, 0),
        "elo2":    round(elo2, 0),
        "form1":   round(float(r1["form"]), 3),
        "form2":   round(float(r2["form"]), 3),
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
    """Предстоящие матчи с коэффициентами + наш прогноз"""
    try:
        url = (
            f"https://api.the-odds-api.com/v4/sports/tennis/odds/"
            f"?apiKey={ODDS_API_KEY}&regions=eu&markets=h2h"
        )
        r = requests.get(url, timeout=10)
        if r.status_code != 200:
            return []

        matches = []
        for m in r.json():
            if not m.get("bookmakers"):
                continue

            home = m["home_team"]
            away = m["away_team"]

            # Коэффициенты
            bk       = m["bookmakers"][0]
            outcomes = bk["markets"][0]["outcomes"]
            odds     = {o["name"]: o["price"] for o in outcomes}

            # Наш прогноз
            our_prob  = None
            value_bet = None
            if ratings_df is not None and model_bundle is not None:
                r1 = find_player(home.split()[-1])
                r2 = find_player(away.split()[-1])
                if r1 is not None and r2 is not None:
                    elo1     = get_elo(r1, "hard")
                    elo2     = get_elo(r2, "hard")
                    elo_prob = 1 / (1 + 10 ** (-(elo1 - elo2) / 400))

                    clf, scaler, features = (
                        model_bundle["clf"],
                        model_bundle["scaler"],
                        model_bundle["features"],
                    )
                    X = pd.DataFrame([{
                        "elo_diff":  elo1 - elo2,
                        "form_diff": float(r1["form"]) - float(r2["form"]),
                        "elo_prob":  elo_prob,
                        "rank_diff": 0,
                        "is_clay":   0,
                        "is_grass":  0,
                        "round_num": 4,
                    }])[features]

                    our_prob = round(float(clf.predict_proba(scaler.transform(X))[0][1]), 4)

                    # Проверяем value — наша вероятность vs вероятность букмекера
                    bk_prob_home = round(1 / odds.get(home, 99), 4) if home in odds else None
                    if our_prob and bk_prob_home:
                        diff = our_prob - bk_prob_home
                        if diff > 0.05:
                            value_bet = f"✅ VALUE на {home} (+{diff:.0%})"
                        elif (1 - our_prob) - (1 - bk_prob_home) > 0.05:
                            value_bet = f"✅ VALUE на {away}"
                        else:
                            value_bet = "⚪ Нет value"

            matches.append({
                "home":       home,
                "away":       away,
                "date":       m["commence_time"][:10],
                "time":       m["commence_time"][11:16],
                "odds_home":  odds.get(home),
                "odds_away":  odds.get(away),
                "bookmaker":  bk["title"],
                "our_prob":   our_prob,
                "value_bet":  value_bet,
            })

        return matches

    except Exception as e:
        print(f"Ошибка /upcoming: {e}")
        return []


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
