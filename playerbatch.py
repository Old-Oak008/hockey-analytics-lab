"""NHL batch skater analysis. Probabilities conditional on participation.
Reuses v1.2 probabilities unchanged; no opponent or confirmed lineup adjustment.
"""
import time
from email.utils import parsedate_to_datetime
from requests import HTTPError
import pandas as pd
from players import roster, logs, before, probabilities



class RateLimited(RuntimeError):
    pass


class PacedGet:
    """Session-local response cache and conservative pacing. Stop on 429.
    State survives Streamlit reruns; no requests are made during cooldown.
    """
    def __init__(self, get, state, interval=1.0, clock=time.time, sleep=time.sleep):
        self.get, self.state, self.interval = get, state, interval
        self.clock, self.sleep = clock, sleep
        self.cache = state.setdefault('responses', {})

    def __call__(self, path):
        now = self.clock()
        hit = self.cache.get(path)
        if hit is not None and now-hit[0] < 1800:
            return hit[1]
        if now < self.state.get('blocked_until', 0):
            raise RateLimited('429: limit NHL API; další pokus po skončení pauzy.')
        delay = self.state.get('last_request', 0)+self.interval-now
        if delay > 0:
            self.sleep(delay)
        self.state['last_request'] = self.clock()
        try:
            value = self.get(path)
        except HTTPError as exc:
            if exc.response is None or exc.response.status_code != 429:
                raise
            now = self.clock()
            header = exc.response.headers.get('Retry-After', '')
            try:
                wait = float(header)
            except (TypeError, ValueError):
                try: wait = parsedate_to_datetime(header).timestamp()-now
                except (TypeError, ValueError, OverflowError): wait = 60
            if not pd.notna(wait) or wait == float('inf'): wait = 60
            self.state['blocked_until'] = now+max(60,wait)
            raise RateLimited('429: limit NHL API; stahování pozastaveno.') from exc
        # Only cache valid responses for the particular endpoint.
        valid = isinstance(value,dict) and (isinstance(value.get('gameLog'),list) if '/game-log/' in path else any(isinstance(value.get(k),list) for k in ('forwards','defensemen')))
        if valid:
            self.cache[path] = (self.clock(), value)
            if len(self.cache)>400:
                oldest=min(self.cache,key=lambda k:self.cache[k][0]); del self.cache[oldest]
        return value

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


def batch(get, home, away, day, year, existing=None, progress=None, workers=4, selected_ids=None):
    """Return rows, excluded players and successful newly fetched raw logs.
    Caller controls cache lifetime. Workers never invoke UI or session state.
    """
    existing=existing or {}
    rows=[]; issues=[]; fetched={}
    people,duplicates=candidates(home,away,get(f'/roster/{home}/{year}{year+1}'),get(f'/roster/{away}/{year}{year+1}'))
    if selected_ids is not None:
        chosen=set(selected_ids)
        people=[r for r in people if r[1] in chosen]
        duplicates=[r for r in duplicates if r[1] in chosen]
        missing=chosen-{r[1] for r in people+duplicates}
        if missing: raise ValueError("Vybraná sestava se neshoduje s aktuální soupiskou. Načti soupisky znovu.")
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
    limited=False
    for i,person in enumerate(people,1):
        team,pid,name=person
        base={'Tým':team,'ID':pid,'Hráč':name}
        try:
            if limited and pid not in existing:
                raise RateLimited('429: odloženo do dalšího pokusu.')
            _,item,row,error=process(person)
            if pid not in existing and not item[1]: fetched[pid]=item
            if row is not None: rows.append({**base,**row})
            else: issues.append({**base,'Důvod':error})
        except RateLimited as exc:
            limited=True
            issues.append({**base,'Důvod':str(exc)})
        except Exception as exc:
            issues.append({**base,'Důvod':str(exc)[:300]})
        if progress is not None: progress(i,len(people))
    result=pd.DataFrame(rows)
    if not result.empty: result=result.sort_values(['Tým','Hráč']).reset_index(drop=True)
    errors=pd.DataFrame(issues,columns=['Tým','ID','Hráč','Důvod'])
    if not errors.empty: errors=errors.sort_values(['Tým','Hráč']).reset_index(drop=True)
    return result,errors,fetched
