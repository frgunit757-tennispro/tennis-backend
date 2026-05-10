"""
Tennis Analyzer - сервер
- api-sports.io для матчей и расписания
- stavka.tv для прогнозов капперов
- ML модель для наших прогнозов
- Gemini AI анализ по запросу /analyze
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import numpy as np
import pickle
import requests
import re
from pathlib import Path
from contextlib import asynccontextmanager
from datetime import datetime, timedelta

DATA              = Path(".")
APISPORTS_KEY     = "393b1cec5d46358d928a03b4a9892e44"
APISPORTS_URL     = "https://v1.tennis.api-sports.io"
GEMINI_KEY        = "AIzaSyC1EFzWSM4XRIDS1dcUYSzYlfEiE5yZoiM"
GEMINI_URL        = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={GEMINI_KEY}"

ratings_df   = None
atp_df       = None
model_bundle = None

def load_data():
    global ratings_df, atp_df, model_bundle
    try:
        ratings_df   = pd.read_csv(DATA / "ratings.csv")
        atp_df       = pd.read_csv(DATA / "atp_all.csv", parse_dates=["tourney_date"])
        with open(DATA / "model.pkl", "rb") as f:
            model_bundle = pickle.load(f)
        print(f"OK {len(ratings_df)} players, {len(atp_df):,} matches")
    except Exception as e:
        print(f"Load error: {e}")

@asynccontextmanager
async def lifespan(app):
    load_data()
    yield

app = FastAPI(title="Tennis Analyzer API", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


# ─── HELPERS ───

def find_player(name):
    if ratings_df is None: return None
    exact = ratings_df[ratings_df["player"] == name]
    if not exact.empty: return exact.iloc[0]
    for part in name.split():
        if len(part) > 3:
            partial = ratings_df[ratings_df["player"].str.contains(part, case=False, na=False)]
            if not partial.empty: return partial.iloc[0]
    return None

def get_elo(row, surface):
    col = {"hard":"elo_hard","clay":"elo_clay","grass":"elo_grass"}.get(surface,"elo_hard")
    return float(row[col]) if col in row.index else float(row["elo_hard"])

def predict_winner(r1, r2, surface):
    if model_bundle is None: return None
    elo1 = get_elo(r1, surface)
    elo2 = get_elo(r2, surface)
    elo_prob = 1 / (1 + 10 ** (-(elo1-elo2)/400))
    clf, scaler, features = model_bundle["clf"], model_bundle["scaler"], model_bundle["features"]
    X = pd.DataFrame([{
        "elo_diff": elo1-elo2, "form_diff": float(r1["form"])-float(r2["form"]),
        "elo_prob": elo_prob, "rank_diff": 0,
        "is_clay": int(surface=="clay"), "is_grass": int(surface=="grass"), "round_num": 4,
    }])[features]
    return float(clf.predict_proba(scaler.transform(X))[0][1])

def get_player_stats(player_name):
    if atp_df is None: return {}
    mask = (atp_df["winner_name"].str.contains(player_name, case=False, na=False) |
            atp_df["loser_name"].str.contains(player_name, case=False, na=False))
    matches = atp_df[mask].dropna(subset=["score"]).tail(50)
    total_games, three_set = [], 0
    wins  = int(atp_df[mask & atp_df["winner_name"].str.contains(player_name, case=False, na=False)].shape[0])
    total = int(atp_df[mask].shape[0])
    for _, row in matches.iterrows():
        games, set_count = 0, 0
        for s in str(row.get("score","")).split():
            try:
                p = s.split("-")
                if len(p)==2:
                    games += int(p[0].split("(")[0]) + int(p[1].split("(")[0])
                    set_count += 1
            except: pass
        if games > 0: total_games.append(games)
        if set_count >= 3: three_set += 1
    return {
        "avg_total":     round(np.mean(total_games),1) if total_games else None,
        "three_set_pct": round(three_set/len(matches)*100,1) if matches.shape[0]>0 else None,
        "win_rate":      round(wins/total*100,1) if total>0 else None,
        "matches_total": total,
    }

def get_deep_stats_2024(player_name, surface="clay"):
    if atp_df is None: return {}
    as_winner = atp_df["winner_name"].str.contains(player_name, case=False, na=False)
    as_loser  = atp_df["loser_name"].str.contains(player_name, case=False, na=False)
    recent = atp_df[as_winner | as_loser]
    recent = recent[recent["tourney_date"].dt.year >= 2024].copy()
    if recent.empty: return {}
    wins  = recent[recent["winner_name"].str.contains(player_name, case=False, na=False)]
    losses = recent[recent["loser_name"].str.contains(player_name, case=False, na=False)]
    total = len(recent)
    surface_stats = {}
    for surf in ["Hard","Clay","Grass"]:
        sm = recent[recent["surface"]==surf]
        if len(sm)>0:
            w = sm[sm["winner_name"].str.contains(player_name, case=False, na=False)]
            surface_stats[surf] = {"wins":len(w),"total":len(sm),"win_pct":round(len(w)/len(sm)*100,1)}
    total_games_list, three_set_count = [], 0
    for _, row in recent.iterrows():
        games, set_count = 0, 0
        for s in str(row.get("score","")).split():
            try:
                p = s.split("-")
                if len(p)==2:
                    games += int(p[0].split("(")[0]) + int(p[1].split("(")[0])
                    set_count += 1
            except: pass
        if games>0: total_games_list.append(games)
        if set_count>=3: three_set_count+=1
    srv = {}
    try:
        w_srv = wins[["w_ace","w_df","w_svpt","w_1stIn","w_1stWon","w_bpSaved","w_bpFaced"]].dropna()
        if not w_srv.empty and w_srv["w_svpt"].sum()>0:
            srv = {
                "aces_per_match":  round(w_srv["w_ace"].mean(),1),
                "first_serve_pct": round(w_srv["w_1stIn"].sum()/w_srv["w_svpt"].sum()*100,1),
                "first_serve_won": round(w_srv["w_1stWon"].sum()/w_srv["w_1stIn"].sum()*100,1) if w_srv["w_1stIn"].sum()>0 else None,
                "bp_saved_pct":    round(w_srv["w_bpSaved"].sum()/w_srv["w_bpFaced"].sum()*100,1) if w_srv["w_bpFaced"].sum()>0 else None,
            }
    except: pass
    last10 = []
    for _, row in recent.sort_values("tourney_date").tail(10).iterrows():
        won = player_name.lower() in str(row["winner_name"]).lower()
        opp = str(row["loser_name"] if won else row["winner_name"]).split()[-1]
        last10.append(f"{'W' if won else 'L'} vs {opp} ({row.get('surface','?')[:2]}) {row.get('score','')}")
    surf5 = []
    sf = {"clay":"Clay","hard":"Hard","grass":"Grass"}.get(surface,"Clay")
    for _, row in recent[recent["surface"]==sf].sort_values("tourney_date").tail(5).iterrows():
        won = player_name.lower() in str(row["winner_name"]).lower()
        opp = str(row["loser_name"] if won else row["winner_name"]).split()[-1]
        surf5.append(f"{'W' if won else 'L'} vs {opp} {row.get('score','')}")
    return {
        "total_matches": total, "wins": len(wins), "losses": len(losses),
        "win_pct": round(len(wins)/total*100,1) if total>0 else 0,
        "surface_stats": surface_stats,
        "avg_total_games": round(np.mean(total_games_list),1) if total_games_list else None,
        "three_set_pct": round(three_set_count/total*100,1) if total>0 else None,
        "serve": srv, "last_10": last10, "last_5_surface": surf5,
    }

def analyze_value(our_prob, bk_odds):
    if not our_prob or not bk_odds: return {"has_value":False,"diff":0}
    bk_prob = 1/bk_odds
    diff = our_prob - bk_prob
    return {"has_value":diff>0.05,"diff":round(diff*100,1)}


# ─── API-SPORTS.IO ───

def apisports_get(endpoint, params={}):
    headers = {
        "x-apisports-key": APISPORTS_KEY,
        "x-rapidapi-host": "v1.tennis.api-sports.io"
    }
    try:
        r = requests.get(f"{APISPORTS_URL}/{endpoint}", headers=headers, params=params, timeout=15)
        if r.status_code == 200:
            return r.json()
        print(f"api-sports error {r.status_code}: {r.text[:200]}")
        return None
    except Exception as e:
        print(f"api-sports request error: {e}")
        return None


def get_today_matches():
    """Получаем матчи на сегодня и завтра с api-sports.io"""
    matches = []
    today = datetime.now().strftime("%Y-%m-%d")
    tomorrow = (datetime.now() + timedelta(days=1)).strftime("%Y-%m-%d")

    for date in [today, tomorrow]:
        data = apisports_get("games", {"date": date})
        if not data or not data.get("response"):
            print(f"No matches for {date}")
            continue

        for game in data["response"]:
            try:
                p1 = game["players"]["home"]["name"]
                p2 = game["players"]["away"]["name"]
                tournament = game.get("tournament", {}).get("name", "ATP/WTA")
                country    = game.get("country", {}).get("name", "")
                surface    = game.get("tournament", {}).get("surface", "Hard").lower()
                game_date  = game.get("date", today)[:10]
                game_time  = game.get("time", "")
                status     = game.get("status", {}).get("short", "NS")

                # Пропускаем завершённые матчи
                if status in ["FT", "Fin", "WO"]:
                    continue

                matches.append({
                    "id":         game.get("id"),
                    "home":       p1,
                    "away":       p2,
                    "tournament": tournament,
                    "country":    country,
                    "surface":    surface,
                    "date":       game_date,
                    "time":       game_time[:5] if game_time else "",
                    "status":     status,
                })
            except Exception as e:
                print(f"Game parse error: {e}")
                continue

        print(f"api-sports: {len(matches)} matches for {date}")

    return matches


def get_match_odds(game_id):
    """Получаем коэффициенты для конкретного матча"""
    data = apisports_get("odds", {"game": game_id})
    if not data or not data.get("response"):
        return {}

    odds = {}
    for bk in data["response"]:
        for bet in bk.get("bets", []):
            if bet["name"] == "Winner":
                for val in bet.get("values", []):
                    if val["value"] == "Home":
                        odds["home"] = float(val["odd"])
                    elif val["value"] == "Away":
                        odds["away"] = float(val["odd"])
            elif bet["name"] == "Total Games" and "total_line" not in odds:
                for val in bet.get("values", []):
                    if val["value"].startswith("Over"):
                        odds["total_over"] = float(val["odd"])
                        try: odds["total_line"] = float(val["value"].split()[-1])
                        except: pass
                    elif val["value"].startswith("Under"):
                        odds["total_under"] = float(val["odd"])
        if odds.get("home"): break

    return odds


# ─── STAVKA.TV PARSER ───

def parse_stavka_predictions():
    """Парсим прогнозы капперов с stavka.tv для обогащения анализа"""
    try:
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/120.0.0.0 Safari/537.36",
            "Accept-Language": "ru-RU,ru;q=0.9",
        }
        r = requests.get("https://stavka.tv/predictions/tennis", headers=headers, timeout=12)
        if r.status_code != 200:
            return {}

        from bs4 import BeautifulSoup
        soup = BeautifulSoup(r.text, "html.parser")
        result = {}

        for a in soup.find_all("a", href=re.compile(r"/matches/tennis/")):
            href = a.get("href", "")
            m = re.search(r"/matches/tennis/(\d{2}-\d{2}-\d{4})-(.+?)(?:#|$)", href)
            if not m: continue
            slug = m.group(2)
            text = a.get_text(" ", strip=True)
            coef = re.search(r"(\d+\.\d+)", text)
            if slug not in result:
                result[slug] = {"tips": [], "href": href}
            if coef:
                result[slug]["tips"].append(text[:200])

        print(f"stavka.tv: {len(result)} predictions")
        return result
    except Exception as e:
        print(f"stavka.tv error: {e}")
        return {}


def find_stavka_prediction(home, away, stavka_data):
    """Ищем прогнозы по имени игрока"""
    tips = []
    for slug, data in stavka_data.items():
        slug_lower = slug.lower()
        h_last = home.split()[-1].lower()
        a_last = away.split()[-1].lower()
        if h_last in slug_lower or a_last in slug_lower:
            tips.extend(data.get("tips", []))
    return tips[:3]


# ─── GEMINI ───

def gemini_analyze(home, away, our_prob, avg_total, total_line, three_set_pct,
                   stats1, stats2, value_h2h, value_total,
                   deep1=None, deep2=None, surface="clay",
                   odds_home=None, odds_away=None,
                   total_over=None, total_under=None,
                   stavka_tips=None):
    try:
        prob1 = round(our_prob*100) if our_prob else "?"
        prob2 = 100-prob1 if isinstance(prob1,int) else "?"
        surf_ru = {"clay":"грунт","hard":"хард","grass":"трава"}.get(surface,surface)

        def fmt(name, d):
            if not d: return f"  {name}: нет данных 2024-2026\n"
            sf = {"clay":"Clay","hard":"Hard","grass":"Grass"}.get(surface,"Clay")
            sd = d.get("surface_stats",{}).get(sf,{})
            srv = d.get("serve",{})
            srv_str = f"1я подача {srv.get('first_serve_pct','?')}% | выигрыш {srv.get('first_serve_won','?')}% | эйсы {srv.get('aces_per_match','?')}" if srv else "нет"
            return (
                f"  {name} (2024-2026, матчей:{d.get('total_matches',0)}):\n"
                f"  - W/L: {d.get('wins','?')}/{d.get('losses','?')} ({d.get('win_pct','?')}%)\n"
                f"  - На {surf_ru}: {sd.get('wins','?')}W/{sd.get('total','?')}M ({sd.get('win_pct','?')}%)\n"
                f"  - Avg тотал: {d.get('avg_total_games','?')} | 3 сета: {d.get('three_set_pct','?')}%\n"
                f"  - Подача: {srv_str}\n"
                f"  - Последние 10: {' | '.join(d.get('last_10',[]))}\n"
                f"  - Последние 5 на {surf_ru}: {' | '.join(d.get('last_5_surface',[]))}\n"
            )

        avg_t1 = (deep1 or {}).get("avg_total_games")
        avg_t2 = (deep2 or {}).get("avg_total_games")
        avg_best = round((avg_t1+avg_t2)/2,1) if avg_t1 and avg_t2 else avg_total
        three1 = (deep1 or {}).get("three_set_pct")
        three2 = (deep2 or {}).get("three_set_pct")
        three_best = round((three1+three2)/2,1) if three1 and three2 else three_set_pct

        stavka_str = ""
        if stavka_tips:
            stavka_str = "\n-- ПРОГНОЗЫ КАППЕРОВ (stavka.tv) --\n" + "\n".join(f"• {t}" for t in stavka_tips)

        prompt = f"""Ты профессиональный теннисный аналитик. Используй реальную статистику ATP 2024-2026.

МАТЧ: {home} vs {away} | Покрытие: {surf_ru.upper()}

-- СТАТИСТИКА ATP 2024-2026 --
{fmt(home, deep1 or {})}
{fmt(away, deep2 or {})}
-- ML-МОДЕЛЬ --
- Вероятность: {home} {prob1}% | {away} {prob2}%
- Avg тотал: {avg_best} | % в 3 сета: {three_best}%

-- КОЭФФИЦИЕНТЫ БУКМЕКЕРА --
- {home}: {odds_home or '?'} | {away}: {odds_away or '?'}
- Тотал {total_line or '?'}: О {total_over or '?'} / У {total_under or '?'}

-- НАШ АНАЛИЗ --
- Исход: {value_h2h or 'нет данных'}
- Тотал: {value_total or 'нет данных'}
{stavka_str}

Дай краткий анализ на русском по 3 критериям. Опирайся на статистику.

ФОРМАТ:
🏆 Исход: [кто победит и почему, есть ли value у букмекера?]
📊 Тотал: [больше/меньше {total_line or '?'} и почему]
🎯 Сет аутсайдера: [возьмёт ли хотя бы 1 сет]
✅ Итог: [одна чёткая рекомендация]

Не финансовый совет."""

        body = {"contents":[{"parts":[{"text":prompt}]}],"generationConfig":{"temperature":0.7,"maxOutputTokens":600}}
        for attempt in range(2):
            try:
                r = requests.post(GEMINI_URL, json=body, timeout=45)
                if r.status_code == 200:
                    text = r.json()["candidates"][0]["content"]["parts"][0]["text"].strip()
                    print(f"Gemini OK: {len(text)} chars")
                    return text
                else:
                    print(f"Gemini HTTP {r.status_code}: {r.text[:200]}")
            except requests.exceptions.Timeout:
                print(f"Gemini timeout attempt {attempt+1}")
            except Exception as ex:
                print(f"Gemini error: {ex}")
                break
        return None
    except Exception as e:
        print(f"Gemini fatal: {e}")
        return None


# ─── ENDPOINTS ───

@app.api_route("/", methods=["GET","HEAD"])
def root():
    return {"status":"ok","service":"Tennis Analyzer API"}

@app.get("/players")
def get_players():
    if ratings_df is None: raise HTTPException(503, "Data not loaded")
    return sorted(ratings_df[ratings_df["matches_total"]>=20]["player"].tolist())

@app.get("/ratings")
def get_ratings(surface: str = "hard", limit: int = 30):
    if ratings_df is None: raise HTTPException(503, "Data not loaded")
    col = {"hard":"elo_hard","clay":"elo_clay","grass":"elo_grass"}.get(surface,"elo_hard")
    df = ratings_df[ratings_df["matches_total"]>=30].copy()
    df = df.sort_values(col, ascending=False).head(limit)
    return [{"player":row["player"],"elo":round(row[col],0),"form":round(row["form"],3),"matches_total":int(row["matches_total"])} for _,row in df.iterrows()]

@app.get("/predict")
def predict_match(p1: str, p2: str, surface: str = "hard"):
    if ratings_df is None or model_bundle is None: raise HTTPException(503, "Data not loaded")
    r1 = find_player(p1)
    r2 = find_player(p2)
    if r1 is None: raise HTTPException(404, f"Player '{p1}' not found")
    if r2 is None: raise HTTPException(404, f"Player '{p2}' not found")
    elo1 = get_elo(r1, surface)
    elo2 = get_elo(r2, surface)
    prob1 = predict_winner(r1, r2, surface)
    stats1 = get_player_stats(r1["player"])
    stats2 = get_player_stats(r2["player"])
    avg_total, three_set = None, None
    if stats1.get("avg_total") and stats2.get("avg_total"):
        avg_total = round((stats1["avg_total"]+stats2["avg_total"])/2,1)
    if stats1.get("three_set_pct") and stats2.get("three_set_pct"):
        three_set = round((stats1["three_set_pct"]+stats2["three_set_pct"])/2,1)
    return {
        "player1":r1["player"],"player2":r2["player"],"surface":surface,
        "prob_p1":round(prob1,4) if prob1 else None,"prob_p2":round(1-prob1,4) if prob1 else None,
        "elo1":round(elo1,0),"elo2":round(elo2,0),
        "form1":round(float(r1["form"]),3),"form2":round(float(r2["form"]),3),
        "avg_total":avg_total,"three_set_pct":three_set,
    }

@app.get("/h2h")
def head_to_head(p1: str, p2: str):
    if atp_df is None: raise HTTPException(503, "Data not loaded")
    r1 = find_player(p1)
    r2 = find_player(p2)
    name1 = r1["player"] if r1 is not None else p1
    name2 = r2["player"] if r2 is not None else p2
    mask = (
        (atp_df["winner_name"].str.contains(p1,case=False,na=False) & atp_df["loser_name"].str.contains(p2,case=False,na=False)) |
        (atp_df["winner_name"].str.contains(p2,case=False,na=False) & atp_df["loser_name"].str.contains(p1,case=False,na=False))
    )
    h2h = atp_df[mask].sort_values("tourney_date")
    wins1 = int(h2h["winner_name"].str.contains(p1,case=False,na=False).sum())
    wins2 = len(h2h)-wins1
    by_surface = {}
    for surf, grp in h2h.groupby("surface"):
        w1 = int(grp["winner_name"].str.contains(p1,case=False,na=False).sum())
        by_surface[surf] = {"wins1":w1,"wins2":len(grp)-w1,"total":len(grp)}
    recent = []
    for _, row in h2h.tail(8).iterrows():
        recent.append({"date":str(row["tourney_date"].date()),"tourney":row.get("tourney_name",""),"surface":row.get("surface",""),"round":row.get("round",""),"winner":row["winner_name"],"score":row.get("score","")})
    return {"player1":name1,"player2":name2,"wins1":wins1,"wins2":wins2,"total":len(h2h),"by_surface":by_surface,"recent":recent}

@app.get("/upcoming")
def get_upcoming():
    """Матчи на сегодня/завтра с api-sports.io + наш ML анализ"""
    all_matches = []

    games = get_today_matches()
    if not games:
        print("No games from api-sports.io")
        return []

    for g in games:
        home    = g["home"]
        away    = g["away"]
        surface = g.get("surface", "clay")

        our_prob, avg_total, three_set_p = None, None, None
        value_h2h, value_total, value_set = None, None, None
        odds_home, odds_away = None, None
        total_line, total_over, total_under = None, None, None

        # Коэффициенты
        if g.get("id"):
            odds = get_match_odds(g["id"])
            odds_home   = odds.get("home")
            odds_away   = odds.get("away")
            total_line  = odds.get("total_line")
            total_over  = odds.get("total_over")
            total_under = odds.get("total_under")

        # ML анализ
        if ratings_df is not None and model_bundle is not None:
            r1 = find_player(home.split()[-1])
            r2 = find_player(away.split()[-1])
            if r1 is not None and r2 is not None:
                our_prob = predict_winner(r1, r2, surface)
                stats1   = get_player_stats(r1["player"])
                stats2   = get_player_stats(r2["player"])
                deep1    = get_deep_stats_2024(r1["player"], surface)
                deep2    = get_deep_stats_2024(r2["player"], surface)
                avg_t1 = deep1.get("avg_total_games")
                avg_t2 = deep2.get("avg_total_games")
                if avg_t1 and avg_t2: avg_total = round((avg_t1+avg_t2)/2,1)
                elif stats1.get("avg_total") and stats2.get("avg_total"): avg_total = round((stats1["avg_total"]+stats2["avg_total"])/2,1)
                three1 = deep1.get("three_set_pct")
                three2 = deep2.get("three_set_pct")
                if three1 and three2: three_set_p = round((three1+three2)/2,1)
                elif stats1.get("three_set_pct") and stats2.get("three_set_pct"): three_set_p = round((stats1["three_set_pct"]+stats2["three_set_pct"])/2,1)
                if our_prob and odds_home:
                    v = analyze_value(our_prob, odds_home)
                    if v["has_value"]: value_h2h = f"✅ {home.split()[-1]} (+{v['diff']}%)"
                    elif odds_away and analyze_value(1-our_prob,odds_away)["has_value"]:
                        v2 = analyze_value(1-our_prob,odds_away)
                        value_h2h = f"✅ {away.split()[-1]} (+{v2['diff']}%)"
                    else: value_h2h = "⚪ Нет value"
                if avg_total and total_line:
                    value_total = f"📊 БОЛЬШЕ {total_line} (avg:{avg_total})" if avg_total>float(total_line) else f"📊 МЕНЬШЕ {total_line} (avg:{avg_total})"
                if three_set_p is not None:
                    value_set = f"🎾 3 сета ({three_set_p}%)" if three_set_p>50 else f"🎾 2 сета ({100-three_set_p:.0f}%)"

        all_matches.append({
            "home":          home,
            "away":          away,
            "date":          g["date"],
            "time":          g["time"],
            "bookmaker":     "api-sports.io",
            "tournament":    g["tournament"],
            "country":       g.get("country",""),
            "surface":       surface,
            "game_id":       g.get("id"),
            "odds_home":     odds_home,
            "odds_away":     odds_away,
            "total_line":    total_line,
            "total_over":    total_over,
            "total_under":   total_under,
            "our_prob":      round(our_prob,4) if our_prob else None,
            "avg_total":     avg_total,
            "three_set_pct": three_set_p,
            "value_h2h":     value_h2h,
            "value_total":   value_total,
            "value_set":     value_set,
            "ai_analysis":   None,
        })

    print(f"Returning {len(all_matches)} matches")
    return all_matches


@app.get("/analyze")
def analyze_match(
    home: str, away: str, surface: str = "clay",
    odds_home: float = None, odds_away: float = None,
    total_line: float = None, total_over: float = None,
    total_under: float = None
):
    """Gemini AI анализ — вызывается только когда пользователь открывает матч"""
    our_prob, avg_total, three_set_p = None, None, None
    value_h2h, value_total = None, None
    stats1, stats2, deep1, deep2 = {}, {}, {}, {}

    if ratings_df is not None and model_bundle is not None:
        r1 = find_player(home.split()[-1])
        r2 = find_player(away.split()[-1])
        if r1 is not None and r2 is not None:
            our_prob = predict_winner(r1, r2, surface)
            stats1   = get_player_stats(r1["player"])
            stats2   = get_player_stats(r2["player"])
            deep1    = get_deep_stats_2024(r1["player"], surface)
            deep2    = get_deep_stats_2024(r2["player"], surface)
            avg_t1 = deep1.get("avg_total_games")
            avg_t2 = deep2.get("avg_total_games")
            if avg_t1 and avg_t2: avg_total = round((avg_t1+avg_t2)/2,1)
            elif stats1.get("avg_total") and stats2.get("avg_total"): avg_total = round((stats1["avg_total"]+stats2["avg_total"])/2,1)
            three1 = deep1.get("three_set_pct")
            three2 = deep2.get("three_set_pct")
            if three1 and three2: three_set_p = round((three1+three2)/2,1)
            if our_prob and odds_home:
                v = analyze_value(our_prob, odds_home)
                if v["has_value"]: value_h2h = f"✅ {home.split()[-1]} (+{v['diff']}%)"
                elif odds_away and analyze_value(1-our_prob,odds_away)["has_value"]:
                    v2 = analyze_value(1-our_prob,odds_away)
                    value_h2h = f"✅ {away.split()[-1]} (+{v2['diff']}%)"
                else: value_h2h = "⚪ Нет value"
            if avg_total and total_line:
                value_total = f"📊 БОЛЬШЕ {total_line} (avg:{avg_total})" if avg_total>float(total_line) else f"📊 МЕНЬШЕ {total_line} (avg:{avg_total})"

    # Прогнозы с stavka.tv
    stavka_data = parse_stavka_predictions()
    stavka_tips = find_stavka_prediction(home, away, stavka_data)

    ai_text = gemini_analyze(
        home, away, our_prob, avg_total, total_line, three_set_p,
        stats1, stats2, value_h2h, value_total,
        deep1=deep1, deep2=deep2, surface=surface,
        odds_home=odds_home, odds_away=odds_away,
        total_over=total_over, total_under=total_under,
        stavka_tips=stavka_tips
    )
    return {
        "ai_analysis":   ai_text,
        "our_prob":      round(our_prob,4) if our_prob else None,
        "avg_total":     avg_total,
        "three_set_pct": three_set_p,
        "value_h2h":     value_h2h,
        "value_total":   value_total,
        "stavka_tips":   stavka_tips,
    }


@app.get("/health")
def health():
    return {"status":"ok","players":len(ratings_df) if ratings_df is not None else 0,"matches":len(atp_df) if atp_df is not None else 0}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
