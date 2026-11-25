"""Manual match-specific lineup records. No predictive coefficients."""
import hashlib
import json
import pandas as pd

GROUPS = ('Útok 1','Útok 2','Útok 3','Útok 4','Obrana 1','Obrana 2','Obrana 3','Další hráči','PP1','PP2','Brankář','Absence')

def match_key(league, day, home, away):
    return hashlib.sha256(json.dumps([str(league),str(day),str(home),str(away)],ensure_ascii=False).encode()).hexdigest()[:24]

def validate_selection(available, selected):
    if set(selected) != set(available):
        raise ValueError('Výběr musí obsahovat oba týmy.')
    ids=set()
    for team, values in selected.items():
        current=set(values)
        if not current or not current.issubset(available[team]):
            raise ValueError('Vyber alespoň jednoho dostupného hráče každého týmu.')
        if ids & current:
            raise ValueError('Stejný hráč je vybraný v obou týmech.')
        ids.update(current)
    return frozenset(ids)

def annotate(rows, roles):
    result=rows.copy()
    if result.empty: return result
    result['Formace']=result['ID'].map(lambda x: roles.get(int(x),{}).get('line',''))
    result['Přesilovka']=result['ID'].map(lambda x: roles.get(int(x),{}).get('pp',''))
    return result

def note_table(league,day,home,away,notes,status):
    return pd.DataFrame([{'soutez':league,'datum':str(day),'domaci':home,'hoste':away,'tym':team,'skupina':group,'text':str(notes.get(team,{}).get(group,'')),'stav':status} for team in (home,away) for group in GROUPS if str(notes.get(team,{}).get(group,'')).strip()])
