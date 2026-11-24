"""Experimental ELH adapter. Public schedule, regular season 2026/27 only."""
import re
from datetime import datetime
from html.parser import HTMLParser
import pandas as pd
import requests
URL='https://www.idnes.cz/hokej/databanka/tipsport-extraliga-202627-los.Umli105058'
TEAMS={'K. Vary':'Karlovy Vary','Třinec':'Třinec','Č. Budějovice':'České Budějovice','Ml. Boleslav':'Mladá Boleslav','Plzeň':'Plzeň','Liberec':'Liberec','Vítkovice':'Vítkovice','Kladno':'Kladno','Litvínov':'Litvínov','Kometa':'Kometa Brno','Olomouc':'Olomouc','Sparta':'Sparta Praha','Pardubice':'Pardubice','Hradec Kr.':'Hradec Králové'}
class Rows(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True); self.rows=[]; self.row=None; self.cell=None
    def handle_starttag(self,tag,attrs):
        if tag=='tr': self.row=[]; self.cell=None
        elif tag in ('td','th') and self.row is not None: self.cell=[]
    def handle_data(self,data):
        if self.cell is not None: self.cell.append(data)
    def handle_endtag(self,tag):
        if tag in ('td','th') and self.cell is not None and self.row is not None:
            self.row.append(' '.join(''.join(self.cell).split())); self.cell=None
        elif tag=='tr' and self.row is not None:
            self.rows.append(self.row); self.row=None; self.cell=None

def parse(html):
    parser=Rows(); parser.feed(html); found={}
    for row in parser.rows:
        for i in range(len(row)-3):
            ds,h,a,s=row[i:i+4]
            if not re.fullmatch(r'\d{1,2}\.\s*\d{1,2}\.\s*20\d{2}',ds) or h not in TEAMS or a not in TEAMS: continue
            day=datetime.strptime(re.sub(r'\s+','',ds),'%d.%m.%Y').date()
            if not (datetime(2026,9,1).date()<=day<=datetime(2027,3,10).date()) or h==a: continue
            s=re.sub(r'\s+','',s).upper(); key=(day.isoformat(),TEAMS[h],TEAMS[a]); value=None
            m=re.fullmatch(r'(0|[1-9]\d?):(0|[1-9]\d?)([PN]?)',s)
            if m:
                x,y=int(m[1]),int(m[2]); suffix=m[3]
                if max(x,y)<=20 and x!=y:
                    if suffix:
                        if abs(x-y)!=1: raise ValueError('Neplatné skóre po prodloužení.')
                        x=int(m[1])-int(int(m[1])>int(m[2])); y=int(m[2])-int(int(m[2])>int(m[1]))
                    value=(x,y)
            if key in found and found[key]!=value: raise ValueError('Rozporné výsledky v rozpisu.')
            found[key]=value
    # All 14 teams, 52 rounds, 364 fixtures. Refuse partial or changed page.
    if len(found)!=364: raise ValueError(f'Rozpis není úplný: {len(found)} z 364 utkání. Data nebyla přijata.')
    counts={t:0 for t in TEAMS.values()}
    for _,h,a in found: counts[h]+=1; counts[a]+=1
    if set(counts.values())!={52}: raise ValueError('Nesouhlasí počet zápasů týmů.')
    from zoneinfo import ZoneInfo
    today=datetime.now(ZoneInfo('Europe/Prague')).date().isoformat()
    data=[(*k,*v) for k,v in found.items() if v is not None and k[0]<today]
    pending=sum(v is None and k[0]<today for k,v in found.items())
    return pd.DataFrame(data,columns=['date','home','away','hg','ag']),sorted(found),pending

def load_elh():
    r=requests.get(URL,timeout=25,headers={'User-Agent':'HockeyAnalyticsLab/0.3 (personal statistics)'})
    r.raise_for_status(); return parse(r.text)
