"""Score only fresh corrected outputs, retain counts and compare matching prefixes."""
import json,sys,hashlib
from pathlib import Path
sys.path.insert(0,'/home/jhl/model_inference/LongBench/LongBench')
from eval import scorer
ROOT=Path('/home/jhl/model_inference/llm_infer/longbench_outputs_qwen8b_fixed_20260926')
TARGET={'qasper':200,'multifieldqa_en':150,'hotpotqa':200,'2wikimqa':200,'gov_report':200,'qmsum':200,'trec':200,'repobench-p':500}
result={};paired={}
for task,target in TARGET.items():
 runs={}
 for tag in ['full','m64','m80']:
  p=ROOT/tag/task/(task+'.jsonl')
  if not p.exists():continue
  # Read newline-terminated records only while a worker may append.
  data=p.read_bytes();data=data[:data.rfind(b'\n')+1]
  rows=[json.loads(l) for l in data.splitlines() if l.strip()]
  if not rows:continue
  assert len(set(r['index'] for r in rows))==len(rows)
  assert all('stop_reason' in r and 'generated_token_ids' in r for r in rows)
  runs[tag]=rows
  result.setdefault(tag,{})[task]={'n':len(rows),'target':target,'complete':len(rows)==target,'score':scorer(task,[r['pred'] for r in rows],[r['answers'] for r in rows],rows[-1].get('all_classes',[])),'eos_n':sum(r['stop_reason']=='eos' for r in rows),'unclosed_think_n':sum('<think>' in r['pred'] and '</think>' not in r['pred'] for r in rows),'snapshot_sha256':hashlib.sha256(data).hexdigest()}
 if len(runs)==3:
  n=min(len(v) for v in runs.values())
  for i in range(n):
   assert len({v[i]['input_hash'] for v in runs.values()})==1
   assert len({v[i]['prompt_hash'] for v in runs.values()})==1
  paired[task]={'n':n,'target':target,'scores':{tag:scorer(task,[r['pred'] for r in rows[:n]],[r['answers'] for r in rows[:n]],rows[0].get('all_classes',[])) for tag,rows in runs.items()}}
out={'full_runs':result,'matched_prefix_diagnostic_only':paired}
(ROOT/'scores_fixed.json').write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps(out,indent=2))
