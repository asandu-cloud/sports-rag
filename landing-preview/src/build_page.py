"""Builds landing-preview/index.html from page.html and the brand logo. Edit page.html, then run this file.

    .venv/bin/python -B landing-preview/src/build_page.py
"""
import re
from pathlib import Path
here=Path(__file__).resolve().parent
root,sp=here.parents[1],here
src=open(f'{sp}/page.html').read()
logo=open(f'{root}/docs/brand/logo/spix-wordmark-white.svg').read().strip()
logo=re.sub(r'<\?xml[^>]*>\s*','',logo)
# Navy and pink: the name and its ball are pink, the ball's patches navy
logo=logo.replace('#FAFDFA','#FF7399').replace('#111E17','#000831')
def word(cls,uid): return logo.replace('spix-white-ball',uid).replace(' role="img"','',1).replace(' aria-label="spix"','',1).replace('<svg ',f'<svg class="{cls}" aria-hidden="true" ',1)
start=logo.index('<g transform="translate(109.56 -59.41)">')
depth=0
for m in re.finditer(r'<g\b|</g>',logo[start:]):
    depth+=1 if m.group(0)=='<g' else -1
    if depth==0: end=start+m.end(); break
ball=logo[start:end].replace('translate(109.56 -59.41)','translate(17 17)').replace('spix-white-ball','run-ball')
clip=''.join(re.findall(r'<clipPath[^>]*>.*?</clipPath>',logo,re.S)).replace('spix-white-ball','run-ball')
ball_svg=f'<svg viewBox="0 0 34 34" width="22" height="22" focusable="false"><defs>{clip}</defs>{ball}</svg>'
out=src.replace('{{LOGO_HEADER}}',word('word','spix-ball-h')).replace('{{LOGO_FOOTER}}',word('foot-word','spix-ball-f')).replace('{{BALL}}',ball_svg)
open(f'{root}/landing-preview/index.html','w').write(out)
print('index.html',len(out),'{{' in out)
