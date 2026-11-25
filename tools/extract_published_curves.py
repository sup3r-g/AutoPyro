"""Extract published vector paths; run once, pdfplumber is not a runtime dependency."""
import json
from pathlib import Path
import pdfplumber
import numpy as np

import argparse
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('pdf', help='SPBA atlas chapter 13 PDF')
parser.add_argument('output', help='Output published_curves.json')
args = parser.parse_args()
with pdfplumber.open(args.pdf) as pdf:
    p = pdf.pages[3]
    curves = [c for c in p.curves if c['stroking_color'] == (0.2, 1., 1., 0.)]
    output = {}
    for name, top, bottom, labels in [('HI–Tmax', 640, 861, ['III', 'II', 'I']), ('HI–OI', 940, 1165, ['I', 'II', 'III', 'III/IV'])]:
        selected = [c for c in curves if c['x0']<550 and c['top']>top and c['bottom']<bottom and c['height']>10 and any(op[0] in ('c','v','y') for op in c['path'])]
        if len(selected) != len(labels):
            raise ValueError(f"Unexpected vector paths for {name}: {len(selected)}")
        entries = []
        for label,c in zip(labels, selected):
            points=[]
            for op in c['path']:
                if op[0]=='m':
                    current=np.array(op[1]); points.append(current)
                else:
                    if op[0]=='c': a,b,end=map(np.array,op[1:])
                    elif op[0]=='v': a=current; b,end=map(np.array,op[1:])
                    elif op[0]=='y': a,end=map(np.array,op[1:]); b=end
                    else: raise ValueError(op)
                    for t in np.linspace(0,1,33)[1:]: points.append((1-t)**3*current+3*(1-t)**2*t*a+3*(1-t)*t*t*b+t**3*end)
                    current=end
            physical=[]
            for x,y in points:
                if name=='HI–Tmax': physical.append([round(400+(y-631.755)/242.94*120,5),round((x-211.944)/334.449*1000,5)])
                else: physical.append([round((x-211.944)/334.449*300,5),round((1167.203-y)/(1167.203-925.713)*900,5)])
            entries.append(dict(id=label,label='Тип '+label,coordinates=physical,vector_path=c['path']))
        output[name]=entries
    Path(args.output).write_text(json.dumps(output,ensure_ascii=False,indent=2),encoding='utf-8')
    print({k:[e['id'] for e in v] for k,v in output.items()})
