"""
Tennis Analyzer - Финальный сервер
- Рейтинги игроков по Elo
- Прогноз матча (ML модель)
- H2H статистика
- Матчи с коэффициентами букмекеров (the-odds-api)
- 3 критерия: исход, тотал, сеты
- Gemini ИИ анализ — только по запросу /analyze
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import numpy as np
import pickle
import requests
from pathlib import Path
from contextlib import asynccontextmanager
from bs4 import BeautifulSoup
import re
from datetime import datetime, date

DATA         = Path(".")
ODDS_API_KEY = "7f2d9a6e51688c0e68bce9abca2876ba"
GEMINI_KEY   = "AIzaSyC1EFzWSM4XRIDS1dcUYSzYlfEiE5yZoiM"
GEMINI_URL   = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={GEMINI_KEY}"

TENNIS_SPORTS = ["tennis", "tennis_atp_italian_open", "tennis_wta_italian_open"]

def get_active_tennis_sports():
    """Динамически получаем все активные теннисные турниры из API"""
    try:
        r = requests.get(
            f"https://api.the-odds-api.com/v4/sports/?apiKey={ODDS_API_KEY}",
            timeout=10
        )
        if r.status_code == 200:
            all_sports = r.json()
            tennis = [s["key"] for s in all_sports if s.get("group") == "Tennis" and s.get("active") and not s.get("has_outrights")]
            print(f"Active tennis sports: {tennis}")
            return tennis if tennis else TENNIS_SPORTS
    except Exception as e:
        print(f"Error fetching sports: {e}")
    return TENNIS_SPORTS



def parse_stavka_tv():
    """Парсим матчи и прогнозы с stavka.tv/predictions/tennis"""
    matches = {}
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml',
            'Accept-Language': 'ru-RU,ru;q=0.9',
        }
        r = requests.get('https://stavka.tv/predictions/tennis', headers=headers, timeout=15)
        if r.status_code != 200:
            print(f"stavka.tv error: {r.status_code}")
            return {}

        soup = BeautifulSoup(r.text, 'html.parser')
        tips = soup.find_all('a', href=re.compile(r'/matches/tennis/'))

        seen = set()
        for tip in tips:
            href = tip.get('href', '')
            match = re.search(r'/matches/tennis/(\d{2}-\d{2}-\d{4})-(.+?)(?:#|$)', href)
            if not match:
                continue
            date_str = match.group(1)
            slug = match.group(2)
            key = slug

            if key in seen:
                continue
            seen.add(key)

            # Парсим имена игроков из slug
            parts = slug.split('-')
            # Ищем разделитель между игроками
            mid = len(parts) // 2
            # Пробуем угадать разделение
            player1 = ' '.join(p.capitalize() for p in parts[:mid])
            player2 = ' '.join(p.capitalize() for p in parts[mid:])

            # Берём текст из ссылки
            text = tip.get_text(separator=' ', strip=True)
            # Ищем коэффициент
            coef_match = re.search(r'(\d+\.\d+)', text)
            coef = float(coef_match.group(1)) if coef_match else None

            # Ищем тип ставки
            bet_type = None
            for bt in ['ПОБЕДА 1', 'ПОБЕДА 2', 'ТОТАЛ БОЛЬШЕ', 'ТОТАЛ МЕНЬШЕ', 'ФОРА']:
                if bt in text.upper():
                    bet_type = bt
                    break

            if key not in matches:
                matches[key] = {
                    'slug': slug,
                    'date': date_str,
                    'href': href,
                    'tips': [],
                    'player1_raw': player1,
                    'player2_raw': player2,
                }

            if coef and bet_type:
                matches[key]['tips'].append({
                    'type': bet_type,
                    'coef': coef,
                    'text': text[:200],
                })

        print(f"stavka.tv: найдено {len(matches)} матчей")
        return matches

    except Exception as e:
        print(f"stavka.tv parse error: {e}")
        return {}


def parse_match_page(slug: str):
    """Парсим страницу конкретного матча на stavka.tv для получения коэффициентов"""
    try:
        today = datetime.now()
        date_str = today.strftime('%d-%m-%Y')
        url = f"https://stavka.tv/matches/tennis/{date_str}-{slug}"
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
            'Accept-Language': 'ru-RU,ru;q=0.9',
        }
        r = requests.get(url, headers=headers, timeout=10)
        if r.status_code != 200:
            return {}

        soup = BeautifulSoup(r.text, 'html.parser')
        result = {}

        # Ищем коэффициенты P1/P2
        odds_blocks = soup.find_all(text=re.compile(r'^\d+\.\d+$'))
        odds = [float(o.strip()) for o in odds_blocks if 1.01 < float(o.strip()) < 50]
        if len(odds) >= 2:
            result['odds_home'] = odds[0]
            result['odds_away'] = odds[1]

        # Ищем время матча
        time_match = soup.find(text=re.compile(r'\d{2}:\d{2}'))
        if time_match:
            result['time'] = time_match.strip()[:5]

        # Собираем все прогнозы капперов
        all_tips = []
        tip_blocks = soup.find_all('p')
        for p in tip_blocks:
            text = p.get_text(strip=True)
            if len(text) > 20 and len(text) < 500:
                all_tips.append(text)
        result['capper_tips'] = all_tips[:5]

        return result
    except Exception as e:
        print(f"Match page parse error: {e}")
        return {}


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
    wins   = recent[recent["winner_name"].str.contains(player_name, case=False, na=False)]
    losses = recent[recent["loser_name"].str.contains(player_name, case=False, na=False)]
    total  = len(recent)
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
    return {"has_value":diff>0.05,"diff":round(diff*100,1),"our_prob_pct":round(our_prob*100,1),"bk_prob_pct":round(bk_prob*100,1)}


def gemini_analyze(home, away, our_prob, avg_total, total_line, three_set_pct,
                   stats1, stats2, value_h2h, value_total,
                   deep1=None, deep2=None, surface="clay",
                   odds_home=None, odds_away=None, total_over=None, total_under=None):
    try:
        prob1 = round(our_prob*100) if our_prob else "?"
        prob2 = 100-prob1 if isinstance(prob1,int) else "?"
        surf_ru = {"clay":"грунт","hard":"хард","grass":"трава"}.get(surface,surface)

        def fmt(name, d):
            if not d: return f"  {name}: нет данных 2024-2026\n"
            sf = {"clay":"Clay","hard":"Hard","grass":"Grass"}.get(surface,"Clay")
            sd = d.get("surface_stats",{}).get(sf,{})
            srv = d.get("serve",{})
            srv_str = (f"1я подача {srv.get('first_serve_pct','?')}% | выигрыш {srv.get('first_serve_won','?')}% | эйсы {srv.get('aces_per_match','?')} | БП спасено {srv.get('bp_saved_pct','?')}%") if srv else "нет"
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

        prompt = f"""Ты профессиональный теннисный аналитик. Используй реальную статистику ATP 2024-2026.

МАТЧ: {home} vs {away} | Покрытие: {surf_ru.upper()}

-- СТАТИСТИКА ATP 2024-2026 --
{fmt(home, deep1 or {})}
{fmt(away, deep2 or {})}
-- ML-МОДЕЛЬ --
- Вероятность: {home} {prob1}% | {away} {prob2}%
- Avg тотал: {avg_best} | % в 3 сета: {three_best}%

-- КОЭФФИЦИЕНТЫ --
- {home}: {odds_home or '?'} | {away}: {odds_away or '?'}
- Тотал {total_line or '?'}: O {total_over or '?'} / U {total_under or '?'}

-- АНАЛИЗ --
- Исход: {value_h2h or 'нет данных'}
- Тотал: {value_total or 'нет данных'}

Дай краткий анализ по 3 критериям на русском языке.

ФОРМАТ:
🏆 Исход: [кто победит и почему, есть ли value у букмекера?]
📊 Тотал: [больше/меньше {total_line or '?'} и почему]
🎯 Сет аутсайдера: [возьмёт ли хотя бы 1 сет]
✅ Итог: [одна чёткая рекомендация]

Не финансовый совет."""

        body = {"contents":[{"parts":[{"text":prompt}]}],"generationConfig":{"temperature":0.6,"maxOutputTokens":500}}
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
    """Матчи с stavka.tv — парсим прогнозы капперов + наш ML анализ"""
    all_matches = []

    try:
        stavka_matches = parse_stavka_tv()
        today_str = datetime.now().strftime('%d-%m-%Y')
        tomorrow_str = (datetime.now().replace(day=datetime.now().day+1)).strftime('%d-%m-%Y')

        for key, m in stavka_matches.items():
            # Берём только сегодняшние и завтрашние матчи
            if m['date'] not in [today_str, tomorrow_str]:
                try:
                    # Проверяем дату
                    match_date = datetime.strptime(m['date'], '%d-%m-%Y')
                    if abs((match_date - datetime.now()).days) > 2:
                        continue
                except:
                    continue

            tips = m.get('tips', [])

            # Извлекаем коэффициенты из прогнозов
            odds_home, odds_away, total_line, total_over, total_under = None, None, None, None, None
            capper_tips_text = []

            for tip in tips:
                coef = tip.get('coef')
                btype = tip.get('type', '')
                text = tip.get('text', '')
                if coef:
                    if 'ПОБЕДА 1' in btype and not odds_home:
                        odds_home = coef
                    elif 'ПОБЕДА 2' in btype and not odds_away:
                        odds_away = coef
                    elif 'ТОТАЛ БОЛЬШЕ' in btype and not total_over:
                        total_over = coef
                        import re as _re
                        tl = _re.search(r'\((\d+\.?\d*)\)', text)
                        if tl: total_line = float(tl.group(1))
                    elif 'ТОТАЛ МЕНЬШЕ' in btype and not total_under:
                        total_under = coef
                if text:
                    capper_tips_text.append(f"[{btype} @{coef}] {text[:120]}")

            # Определяем имена игроков из slug
            slug = m['slug']
            parts = slug.replace('-', ' ').split()
            # Пробуем найти игроков в нашей базе
            home_name, away_name = None, None
            if ratings_df is not None:
                for i in range(1, len(parts)):
                    candidate1 = ' '.join(parts[:i]).title()
                    candidate2 = ' '.join(parts[i:]).title()
                    r1 = find_player(candidate1.split()[-1])
                    r2 = find_player(candidate2.split()[-1])
                    if r1 is not None and r2 is not None:
                        home_name = r1['player']
                        away_name = r2['player']
                        break

            if not home_name:
                # Fallback — делим slug пополам
                mid = len(parts) // 2
                home_name = ' '.join(p.capitalize() for p in parts[:mid])
                away_name = ' '.join(p.capitalize() for p in parts[mid:])

            # ML анализ
            our_prob, avg_total, three_set_p = None, None, None
            value_h2h, value_total, value_set = None, None, None

            if ratings_df is not None and model_bundle is not None:
                r1 = find_player(home_name.split()[-1])
                r2 = find_player(away_name.split()[-1])
                if r1 is not None and r2 is not None:
                    our_prob = predict_winner(r1, r2, "clay")
                    stats1 = get_player_stats(r1["player"])
                    stats2 = get_player_stats(r2["player"])
                    deep1 = get_deep_stats_2024(r1["player"], "clay")
                    deep2 = get_deep_stats_2024(r2["player"], "clay")
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
                        if v["has_value"]: value_h2h = f"ok {home_name.split()[-1]} (+{v['diff']}%)"
                        elif odds_away and analyze_value(1-our_prob,odds_away)["has_value"]:
                            v2 = analyze_value(1-our_prob,odds_away)
                            value_h2h = f"ok {away_name.split()[-1]} (+{v2['diff']}%)"
                        else: value_h2h = "no value"
                    if avg_total and total_line:
                        value_total = f"BOLSHE {total_line} (avg:{avg_total})" if avg_total>float(total_line) else f"MENSHE {total_line} (avg:{avg_total})"
                    if three_set_p is not None:
                        value_set = f"3 sets ({three_set_p}%)" if three_set_p>50 else f"2 sets ({100-three_set_p:.0f}%)"

            all_matches.append({
                "home": home_name,
                "away": away_name,
                "date": m['date'],
                "time": "",
                "bookmaker": "stavka.tv",
                "tournament": "ATP/WTA",
                "odds_home": odds_home,
                "odds_away": odds_away,
                "total_line": total_line,
                "total_over": total_over,
                "total_under": total_under,
                "spread_home": None,
                "spread_away": None,
                "our_prob": round(our_prob,4) if our_prob else None,
                "avg_total": avg_total,
                "three_set_pct": three_set_p,
                "value_h2h": value_h2h,
                "value_total": value_total,
                "value_set": value_set,
                "capper_tips": capper_tips_text[:3],
                "stavka_url": f"https://stavka.tv{m['href']}",
                "ai_analysis": None,
            })

    except Exception as e:
        print(f"Upcoming error: {e}")

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
                if v["has_value"]: value_h2h = f"ok {home.split()[-1]} (+{v['diff']}%)"
                elif odds_away and analyze_value(1-our_prob,odds_away)["has_value"]:
                    v2 = analyze_value(1-our_prob,odds_away)
                    value_h2h = f"ok {away.split()[-1]} (+{v2['diff']}%)"
                else: value_h2h = "no value"
            if avg_total and total_line:
                value_total = f"BOLSHE {total_line} (avg:{avg_total})" if avg_total>float(total_line) else f"MENSHE {total_line} (avg:{avg_total})"

    ai_text = gemini_analyze(
        home, away, our_prob, avg_total, total_line, three_set_p,
        stats1, stats2, value_h2h, value_total,
        deep1=deep1, deep2=deep2, surface=surface,
        odds_home=odds_home, odds_away=odds_away,
        total_over=total_over, total_under=total_under
    )
    return {
        "ai_analysis":   ai_text,
        "our_prob":      round(our_prob,4) if our_prob else None,
        "avg_total":     avg_total,
        "three_set_pct": three_set_p,
        "value_h2h":     value_h2h,
        "value_total":   value_total,
    }


@app.get("/health")
def health():
    return {"status":"ok","players":len(ratings_df) if ratings_df is not None else 0,"matches":len(atp_df) if atp_df is not None else 0}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
