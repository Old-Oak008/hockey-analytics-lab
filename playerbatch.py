"""NHL batch skater analysis. Probabilities conditional on participation.
Reuses v1.2 probabilities unchanged; no opponent or confirmed lineup adjustment.
"""
from concurrent.futures import ThreadPoolExecutor, as_completed
from requests import HTTPError
import pandas as pd
from players import roster, logs, before, probabilities


def fixtures_for_day(payload, day):
    target = pd.Timestamp(day).date().isoformat()
    pairs = []
    for week in payload['gameWeek']:
        if week['date'] != target:
            continue
        for g in week['games']:
            if g.get('gameType') == 2:
                pair = (g['homeTeam']['abbrev'], g['awayTeam']['abbrev'])
                if pair not in pairs:
                    pairs.append(pair)
    return pairs


def candidates(home, away, home_payload, away_payload):
    if home == away:
        raise ValueError('Vyber dva různé týmy.')
    rows = [(team, pid, name) for team, payload in [(home,home_payload),(away,away_payload)]
            for pid,name in roster(payload).items()]
    counts = {}
    for _,pid,_ in rows:
        counts[pid] = counts.get(pid,0)+1
    return [r for r in rows if counts[r[1]] == 1], [r for r in rows if counts[r[1]] > 1]


def fetch_logs(get, pid, year):
    payloads=[]; missing=[]
    for y in (year-1,year):
        try:
            payloads.append(get(f'/player/{pid}/game-log/{y}{y+1}/2'))
        except HTTPError as exc:
            if exc.response is not None and exc.response.status_code==404:
                missing.append(str(y))
            else:
                raise
    return logs(payloads), missing


def analyze_player(data, day):
    past = before(data, day)
    p = probabilities(past)
    age = (pd.Timestamp(day).normalize() - past.date.max()).days
    last = past.tail(20)
    return {'Zápasy':len(past),'Bod ANO':p['points'][0], 'Bod NE':1-p['points'][0],
            'Gól ANO':p['goals'][0], 'Body/zápas L5':past.tail(5).points.mean(),
            'Body/zápas L10':past.tail(10).points.mean(), 'Body/zápas L20':last.points.mean(),
            'Čas na ledě L20 (min)':last.toi.mean()/60,
            'Poslední zápas':past.date.max().date().isoformat(),
            'Stáří (dny)':age,
            'Stav':'Omezená historie' if len(past)<30 or age>21 else '30+ zápasů, aktuální'}


def batch(get, home, away, day, year, existing=None, progress=None, workers=4):
    """Return rows, excluded players and successful newly fetched raw logs.
    Caller controls cache lifetime. Workers never invoke UI or session state.
    """
    existing=existing or {}
    rows=[]; issues=[]; fetched={}
    people,duplicates=candidates(home,away,get(f'/roster/{home}/{year}{year+1}'),get(f'/roster/{away}/{year}{year+1}'))
    for team,pid,name in duplicates:
        issues.append({'Tým':team,'ID':pid,'Hráč':name,'Důvod':'Hráč je v obou sezonních soupiskách; vynechán.'})
    def process(person):
        team,pid,name=person
        item=existing.get(pid)
        if item is None:
            item=fetch_logs(get,pid,year)
        try:
            row=analyze_player(item[0],day)
            if item[1]:
                row['Stav']='Část sezon nedostupná (404); '+row['Stav']
            return person,item,row,None
        except ValueError as exc:
            return person,item,None,str(exc)
    with ThreadPoolExecutor(max_workers=max(1,min(4,workers))) as pool:
        futures={pool.submit(process,p):p for p in people}
        for i,future in enumerate(as_completed(futures),1):
            team,pid,name=futures[future]
            base={'Tým':team,'ID':pid,'Hráč':name}
            try:
                _,item,row,error=future.result()
                # A missing season is retried on the next requested run, not cached.
                if pid not in existing and not item[1]: fetched[pid]=item
                if row is not None: rows.append({**base,**row})
                else: issues.append({**base,'Důvod':error})
            except Exception as exc:
                issues.append({**base,'Důvod':str(exc)[:300]})
            if progress is not None: progress(i,len(people))
    result=pd.DataFrame(rows)
    if not result.empty: result=result.sort_values(['Tým','Hráč']).reset_index(drop=True)
    errors=pd.DataFrame(issues,columns=['Tým','ID','Hráč','Důvod'])
    if not errors.empty: errors=errors.sort_values(['Tým','Hráč']).reset_index(drop=True)
    return result,errors,fetched
