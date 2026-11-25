"""Portable prediction journal and optional private GitHub persistence.
No credentials in code, no public repo writes, no global cache of archive data.
"""
import base64
import io
import re
import numpy as np
import pandas as pd
import requests

COLS=['created_utc','date','home','away','status','model_1x2','model_goals','p1','px','p2','over_4.5','over_5.5','over_6.5','origin']
KEY=['date','home','away','model_1x2','model_goals']
STATUS='předzápasová (API ověřeno)'
LIMIT=700_000

def empty(): return pd.DataFrame(columns=COLS)

def clean(d, imported=False):
    if d.empty: return empty()
    if not set(COLS[:-1])<=set(d): raise ValueError('Neplatný formát archivu NHL.')
    r=d.copy()
    if imported or 'origin' not in r: r['origin']='import CSV'
    r=r[COLS].copy()
    for c in ['home','away','status','model_1x2','model_goals','origin']:
        if r[c].isna().any(): raise ValueError('Chybí textová hodnota v archivu.')
        r[c]=r[c].astype(str)
        if not r[c].str.len().between(1,150).all() or r[c].str.contains(r'[\r\n\x00]',regex=True).any() or r[c].str.match(r'^\s*[=+@-]').any(): raise ValueError('Neplatný text v archivu.')
    if not r.home.str.fullmatch(r'[A-Z]{2,3}').all() or not r.away.str.fullmatch(r'[A-Z]{2,3}').all() or (r.home==r.away).any(): raise ValueError('Neplatné týmy NHL.')
    if not r.status.eq(STATUS).all(): raise ValueError('Archiv přijímá pouze záznamy označené jako předzápasové.')
    if not r.origin.isin(['aplikace','import CSV']).all(): raise ValueError('Neznámý původ záznamu.')
    r['created_utc']=pd.to_datetime(r.created_utc,utc=True,errors='raise',format='mixed').map(lambda x:x.isoformat())
    r['date']=pd.to_datetime(r.date,format='%Y-%m-%d',errors='raise').dt.strftime('%Y-%m-%d')
    if r[['created_utc','date']].isna().any().any(): raise ValueError('Chybí datum.')
    cols=COLS[7:13]; r[cols]=r[cols].apply(pd.to_numeric,errors='raise'); v=r[cols].to_numpy(float)
    if not np.isfinite(v).all() or ((v<0)|(v>1)).any() or not np.allclose(v[:,:3].sum(axis=1),1,atol=1e-6,rtol=0) or (np.diff(v[:,3:],axis=1)>1e-9).any(): raise ValueError('Neplatné pravděpodobnosti.')
    return r

def merge(existing,new):
    # Existing entries win; no replacement by a later forecast or imported file.
    a,b=clean(existing),clean(new)
    b=b.assign(_time=pd.to_datetime(b.created_utc,utc=True,format='mixed')).sort_values('_time',kind='stable').drop(columns='_time') if not b.empty else b
    frames=[f for f in (a,b) if not f.empty]
    return pd.concat(frames,ignore_index=True).drop_duplicates(KEY,keep='first').reset_index(drop=True) if frames else empty()

def encode(d):
    raw=clean(d).to_csv(index=False).encode('utf-8')
    if len(raw)>LIMIT: raise ValueError('Archiv dosáhl limitu 700 kB. Stáhni zálohu; další rozšíření úložiště je nutné.')
    return raw

def decode(raw,imported=False):
    if not raw or len(raw)>LIMIT: raise ValueError('Prázdný nebo příliš velký archiv (max. 700 kB).')
    return clean(pd.read_csv(io.BytesIO(raw)),imported=imported)

class PrivateGitHub:
    def __init__(self,repo,token,client=requests):
        if not re.fullmatch(r'[A-Za-z0-9-]+/[A-Za-z0-9_.-]+',repo): raise ValueError('Neplatný název soukromého repozitáře.')
        self.base='https://api.github.com/repos/'+repo; self.url=self.base+'/contents/predictions.csv'; self.client=client
        self.headers={'Authorization':'Bearer '+token,'Accept':'application/vnd.github+json','X-GitHub-Api-Version':'2022-11-28'}
    def _call(self,method,url,**kw):
        try: return self.client.request(method,url,headers=self.headers,timeout=20,allow_redirects=False,**kw)
        except requests.RequestException: raise ValueError('GitHub není dostupný. Záznam zůstává v této relaci.') from None
    def read(self):
        r=self._call('GET',self.base)
        if r.status_code!=200 or r.json().get('private') is not True: raise ValueError('Archiv vyžaduje dostupný soukromý repozitář. Zkontroluj oprávnění a token.')
        r=self._call('GET',self.url)
        if r.status_code==404: return empty(),None
        if r.status_code!=200: raise ValueError(f'GitHub: archiv nelze načíst (HTTP {r.status_code}).')
        j=r.json()
        if j.get('encoding')!='base64' or j.get('size',LIMIT+1)>LIMIT: raise ValueError('Neplatný nebo příliš velký vzdálený archiv.')
        return decode(base64.b64decode(j['content'])),j['sha']
    def sync(self,local):
        remote,sha=self.read(); combined=merge(remote,local)
        if len(combined)==len(remote): return remote
        body={'message':'Update hockey prediction journal','content':base64.b64encode(encode(combined)).decode('ascii')}
        if sha: body['sha']=sha
        r=self._call('PUT',self.url,json=body)
        if r.status_code not in (200,201): raise ValueError(f'Archiv nebyl uložen (HTTP {r.status_code}). Zkus synchronizaci znovu; místní záznamy zůstávají zachovány.')
        return combined
