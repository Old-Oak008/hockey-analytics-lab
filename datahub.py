"""Portable, session-only data snapshots. No pickle, extraction or remote upload."""
import io
import json
import re
from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED, BadZipFile
import pandas as pd
from model import validate

SLOTS = ('mp12_session', 'goalie12_session', 'own14_session', 'extra_elh14_session', 'extra_own14_session', 'extra_nhl14_session')
MAX_RAW = 180_000_000  # conservative budget for Community Cloud
MAX_FILES = 48
SEASON = re.compile(r'20\d{2}20\d{2}')

def season_data(raw, season):
    if not SEASON.fullmatch(season) or int(season[4:]) != int(season[:4])+1:
        raise ValueError('Neplatná sezona NHL.')
    d = validate(pd.read_csv(io.BytesIO(raw)))
    year = int(season[:4]); start = pd.Timestamp(year,7,1); end = pd.Timestamp(year+1,7,1)
    if d.empty or not d.date.between(start,end,inclusive='left').all():
        raise ValueError(f'Neplatná nebo prázdná historie sezony {season}.')
    if d.duplicated(['date','home','away']).any():
        raise ValueError(f'Rozporné výsledky sezony {season}.')
    return d

def complete(d, season):
    # Only full 32-team, 82-game seasons qualify for automatic skipping.
    if int(season[:4])<2021: return False
    counts = pd.concat([d.home,d.away]).value_counts()
    return len(d)==1312 and len(counts)==32 and counts.eq(82).all() and not d.duplicated(['date','home','away']).any()

def checked_payload(name, raw):
    if not isinstance(name,str) or not name or len(name)>220 or '/' in name or '\\' in name or not name.lower().endswith(('.csv','.zip')):
        raise ValueError('Neplatný název datového souboru.')
    if not isinstance(raw,bytes) or not raw or len(raw)>MAX_RAW: raise ValueError('Prázdný nebo příliš velký datový soubor.')
    if name.lower().endswith('.zip'):
        with ZipFile(io.BytesIO(raw)) as z:
            entries=z.infolist()
            if len(entries)>500 or sum(i.file_size for i in entries)>350_000_000: raise ValueError('Vnitřní ZIP je příliš velký.')
            if not any(i.filename.lower().endswith('.csv') and not i.is_dir() for i in entries): raise ValueError('Vnitřní ZIP neobsahuje CSV.')
    return name,raw

def build_backup(seasons, slots):
    # Keep original MoneyPuck bytes, including situations used by other models.
    manifest={'format':'HockeyAnalyticsLab-data-v1','seasons':[], 'uploads':[]}
    entries=[]; size=0
    for season, raw in sorted(seasons.items()):
        season_data(raw,season); path=f'nhl/{season}.csv'
        manifest['seasons'].append({'season':season,'path':path}); entries.append((path,raw)); size+=len(raw)
    for slot in SLOTS:
        for i,(name,raw) in enumerate(slots.get(slot,())):
            name,raw=checked_payload(name,raw)
            if slot not in ('mp12_session','goalie12_session') and not name.lower().endswith('.csv'): raise ValueError('Vlastní ligová data musejí být CSV.')
            path=f'uploads/{slot}/{i}'; manifest['uploads'].append({'slot':slot,'name':name,'path':path}); entries.append((path,raw)); size+=len(raw)
    if not entries: raise ValueError('Nejsou načtena žádná data k záloze.')
    if size>MAX_RAW or len(entries)>MAX_FILES: raise ValueError('Záloha je příliš velká (max. 180 MB vstupních souborů / 48 souborů). Odeber nepotřebné soubory.')
    out=io.BytesIO()
    with ZipFile(out,'w',ZIP_DEFLATED,compresslevel=1) as z:
        z.writestr('manifest.json',json.dumps(manifest,ensure_ascii=False));
        for path,raw in entries: z.writestr(path,raw)
    return out.getvalue()

def restore_backup(raw):
    if len(raw)>MAX_RAW: raise ValueError('Záloha je příliš velká.')
    with ZipFile(io.BytesIO(raw)) as z:
        infos=z.infolist(); names=[i.filename for i in infos]
        if len(names)!=len(set(names)) or len(names)>MAX_FILES+1 or sum(i.file_size for i in infos)>MAX_RAW+65536:
            raise ValueError('Neplatná nebo příliš velká záloha.')
        if 'manifest.json' not in names or z.getinfo('manifest.json').file_size>65536: raise ValueError('Nejde o zálohu Hockey Analytics Lab.')
        m=json.loads(z.read('manifest.json'))
        if m.get('format')!='HockeyAnalyticsLab-data-v1': raise ValueError('Neznámý formát zálohy.')
        used={'manifest.json'}; seasons={}; slots={}
        for rec in m['seasons']:
            season,path=rec['season'],rec['path']
            if season in seasons or path in used or path!=f'nhl/{season}.csv': raise ValueError('Duplicitní nebo neplatná sezona.')
            value=z.read(path); season_data(value,season); seasons[season]=value; used.add(path)
        for rec in m['uploads']:
            slot,path=rec['slot'],rec['path']; name=rec['name']
            if slot not in SLOTS or path in used or not re.fullmatch(r'uploads/'+re.escape(slot)+r'/\d+',path): raise ValueError('Neplatný soubor zálohy.')
            if slot in ('mp12_session','own14_session') and slot in slots: raise ValueError('Více souborů v jednom datovém poli.')
            if slot not in ('mp12_session','goalie12_session') and not name.lower().endswith('.csv'): raise ValueError('Vlastní ligová data musejí být CSV.')
            slots.setdefault(slot,[]).append(checked_payload(name,z.read(path))); used.add(path)
        if used!=set(names) or len(used)==1: raise ValueError('Záloha obsahuje neznámé soubory nebo je prázdná.')
    return seasons,{slot:tuple(files) for slot,files in slots.items()}
