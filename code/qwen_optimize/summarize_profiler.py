import csv,re
from pathlib import Path
out=Path(__file__).parent/'results'
rows=[]
pat=re.compile(r'^\s*(.*?)\s+([\d.]+)%\s+([\d.]+)(us|ms)\s+[\d.]+%\s+[\d.]+(us|ms)\s+[\d.]+us\s+([\d.]+)%\s+([\d.]+)(us|ms)\s+[\d.]+us.*?\s+(\d+)\s*$')
def ms(v,u): return float(v)*(0.001 if u=='us' else 1)
for p in sorted(out.glob('profiler_reference_m*.txt')):
 m=re.search(r'm(\d+)',p.name).group(1)
 for line in p.read_text().splitlines():
  x=pat.match(line)
  if x:
   name=x.group(1).strip(); rows.append({'m':m,'name':name,'calls':int(x.group(8)),'cuda_self_ms':ms(x.group(6),x.group(7)),'cuda_total_ms':ms(x.group(8-1),x.group(9))})
# fallback robust fixed-column parser from whitespace tail
if not rows:
 for p in sorted(out.glob('profiler_reference_m*.txt')):
  m=re.search(r'm(\d+)',p.name).group(1)
  for line in p.read_text().splitlines()[3:]:
   parts=line.split()
   if len(parts)>=15 and parts[-1].isdigit():
    try:
     def val(i): return float(parts[i][:-2])*(.001 if parts[i].endswith('us') else 1)
     rows.append({'m':m,'name':' '.join(parts[:-14]),'calls':int(parts[-1]),'cuda_self_ms':val(7),'cuda_total_ms':val(9)})
    except: pass
for m in ('64','80','96'):
 subset=[r for r in rows if r['m']==m]; total=sum(r['cuda_total_ms'] for r in subset)
 for r in subset: r['percentage']=100*r['cuda_total_ms']/total if total else 0
 subset.sort(key=lambda r:r['cuda_total_ms'],reverse=True)
 for i,r in enumerate(subset,1): r['rank']=i
fields=['m','rank','name','calls','cuda_self_ms','cuda_total_ms','percentage']
with (out/'reference_profiler_stage_summary.csv').open('w',newline='') as f:
 w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(sorted(rows,key=lambda x:(x['m'],x['rank'])))
with (out/'reference_top_cuda_kernels.csv').open('w',newline='') as f:
 w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows([r for r in rows if r['rank']<=15])
print('rows',len(rows))
for m in ('64','80','96'):
 print('\nm',m)
 for r in [x for x in rows if x['m']==m and x['rank']<=10]: print(r['rank'],r['name'],r['calls'],round(r['cuda_self_ms'],3),round(r['cuda_total_ms'],3),round(r['percentage'],2))
