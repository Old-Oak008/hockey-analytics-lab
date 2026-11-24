import requests
import pandas as pd
from model import validate
BASE='https://api-web.nhle.com/v1'
def get(path):
    r=requests.get(BASE+path,timeout=20); r.raise_for_status(); return r.json()
def load(season,progress=lambda x:None):
    teams='ANA BOS BUF CAR CBJ CGY CHI COL DAL DET EDM FLA LAK MIN MTL NJD NSH NYI NYR OTT PHI PIT SEA SJS STL TBL TOR VAN VGK WPG WSH'.split()+(['ARI'] if int(season[:4])<2024 else ['UTA'])
    games={}; errors=[]
    for i,t in enumerate(teams):
        progress((i+1)/len(teams))
        try:
            for g in get(f'/club-schedule-season/{t}/{season}')['games']:
                if g.get('gameType')!=2 or g.get('gameState') not in ('OFF','FINAL'): continue
                h,a=g['homeTeam'],g['awayTeam']; hg,ag=h['score'],a['score']; typ=g.get('gameOutcome',{}).get('lastPeriodType')
                if typ not in ('REG','OT','SO'): continue
                if typ!='REG':
                    if hg>ag: hg-=1
                    elif ag>hg: ag-=1
                games[g['id']]=[g['gameDate'],h['abbrev'],a['abbrev'],hg,ag]
        except (requests.RequestException,KeyError,ValueError) as e: errors.append(f'{t}: {e}')
    if not games: raise ValueError('API nevrátilo žádné výsledky. '+ '; '.join(errors[:2]))
    return validate(pd.DataFrame(games.values(),columns=['date','home','away','hg','ag'])),errors
