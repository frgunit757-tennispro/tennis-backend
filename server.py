"""
Бэкенд для Telegram Mini App
Запускай: python server.py
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import numpy as np
import pickle
from pathlib import Path

app = FastAPI(title="Tennis Analyzer API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─── Загрузка данных ───
DATA = Path("data")
ratings_df: pd.DataFrame = None
atp_df:     pd.DataFrame = None
model_bundle: dict = None

@app.on_event("startup")
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
        print("Убедись что ты запустил Colab и данные в папке data/")


def find_player(name: str) -> pd.Series | None:
    if ratings_df is None:
        return None
    exact = ratings_df[ratings_df["player"] == name]
    if not exact.empty:
        return exact.iloc[0]
    partial = ratings_df[ratings_df["player"].str.contains(name, case=False, na=False)]
    if not partial.empty:
        return partial.iloc[0]
    return None


# ─── Эндпоинты ───

@app.get("/players")
def get_players():
    """Список всех игроков (для автокомплита)"""
    if ratings_df is None:
        raise HTTPException(503, "Данные не загружены")
    players = ratings_df[ratings_df["matches_total"] >= 20]["player"].tolist()
    return sorted(players)


@app.get("/ratings")
def get_ratings(surface: str = "hard", limit: int = 30):
    """Топ игроки по выбранному покрытию"""
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
    """Прогноз матча"""
    if ratings_df is None or model_bundle is None:
        raise HTTPException(503, "Данные не загружены")

    r1 = find_player(p1)
    r2 = find_player(p2)

    if r1 is None:
        raise HTTPException(404, f"Игрок '{p1}' не найден")
    if r2 is None:
        raise HTTPException(404, f"Игрок '{p2}' не найден")

    col_map = {"hard": "elo_hard", "clay": "elo_clay", "grass": "elo_grass"}
    col = col_map.get(surface, "elo_hard")

    elo1 = float(r1[col]) if col in r1.index else float(r1["elo_hard"])
    elo2 = float(r2[col]) if col in r2.index else float(r2["elo_hard"])

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
        "player1":  r1["player"],
        "player2":  r2["player"],
        "surface":  surface,
        "prob_p1":  round(prob1, 4),
        "prob_p2":  round(1 - prob1, 4),
        "elo1":     round(elo1, 0),
        "elo2":     round(elo2, 0),
        "form1":    round(float(r1["form"]), 3),
        "form2":    round(float(r2["form"]), 3),
    }


@app.get("/h2h")
def head_to_head(p1: str, p2: str):
    """История встреч двух игроков"""
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

    # По покрытиям
    by_surface = {}
    for surf, grp in h2h.groupby("surface"):
        w1 = int(grp["winner_name"].str.contains(p1, case=False, na=False).sum())
        by_surface[surf] = {"wins1": w1, "wins2": len(grp)-w1, "total": len(grp)}

    # Последние матчи
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
        "player1":   name1,
        "player2":   name2,
        "wins1":     wins1,
        "wins2":     wins2,
        "total":     len(h2h),
        "by_surface": by_surface,
        "recent":    recent,
    }


@app.get("/health")
def health():
    return {
        "status": "ok",
        "players": len(ratings_df) if ratings_df is not None else 0,
        "matches": len(atp_df) if atp_df is not None else 0,
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
