import sys,json,pathlib,hashlib
sys.path.insert(0,'/home/jhl/model_inference/LongBench/LongBench')
from eval import scorer
root=pathlib.Path('/home/jhl/model_inference/llm_infer')
tasks={'qasper':200,'multifieldqa_en':150,'hotpotqa':200,'2wikimqa':200,'gov_report':200,'qmsum':200,'trec':200,'repobench-p':500}
result={}
for model,folder,methods in [('Qwen3-8B','longbench_outputs_qwen8b_full',['full','Palu','OjaKV','ReCalKV','ours']),('Qwen3-32B','longbench_outputs_32b',['full','ours_m64','ours_m80'])]:
 result[model]={}
 for method in methods:
  out={}
  for task,target in tasks.items():
   p=root/folder/method/(task+'.jsonl') if model=='Qwen3-32B' else root/folder/method/task/(task+'.jsonl')
   if not p.exists(): continue
   raw=p.read_bytes();rows=[json.loads(l) for l in raw.splitlines() if l.strip()]
   if not rows:continue
   out[task]={'n':len(rows),'target':target,'score':scorer(task,[r['pred'] for r in rows],[r['answers'] for r in rows],rows[-1].get('all_classes',[])),'path':str(p),'sha256':hashlib.sha256(raw).hexdigest(),'role_marker_n':sum('assistant\n' in r['pred'] or 'im_end' in r['pred'] for r in rows)}
  result[model][method]=out
print(json.dumps(result,indent=2))
