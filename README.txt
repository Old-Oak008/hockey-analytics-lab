HOCKEY ANALYTICS LAB v0.1
1. Nainstaluj Python 3.12 (64bit) z python.org. Pri instalaci ponech Python Launcher a zaskrtni Add Python to PATH.
2. Rozbal cely ZIP do slozky Dokumenty. Spust Start.bat dvojklikem. Prvni spusteni vyzaduje internet.
3. Otevri http://localhost:8501 v prohlizeci, pokud se neotevre sam.
4. Vyber NHL API, sezonu a klikni Stahnout / aktualizovat NHL. Nebo vyzkousej Rucni scenar.
5. iPhone: stejna domaci Wi-Fi, pocitac musi bezet. V cernem okne najdi IPv4 Address u aktivniho Wi-Fi/Ethernet adapteru. V Safari zadej http://TVOJE-IP:8501 (napr. http://192.168.1.10:8501). Firewall povol pouze pro duveryhodne privatni site. Neotvirej port na routeru, aplikace nema prihlasovani.
6. Ukonceni: Ctrl+C v cernem okne. Pri dalsim spusteni znovu Start.bat.
CSV pro jine ligy: date,home,away,hg,ag; jeden radek na zapas, pouze zakladni hraci doba.
Zdroj NHL: https://api-web.nhle.com/v1/club-schedule-season/{team}/{season}
Model: nezavisly Poisson, domaci/venkovni utok a obrana, vyhlazeni 8 zapasy. Pouze zakladni hraci doba. NHL import odecita rozhodujici gol OT/SO. Bez automatickeho importu ceskych lig, Elo a brankaru. Test bez budoucich dat. Nejde o zaruku vysledku.
