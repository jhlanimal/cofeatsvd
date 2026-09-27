"""Run matched corrected Full/m64/m80 evaluations on GPUs 0--5 only."""
import concurrent.futures,json,os,subprocess,time,threading,hashlib
from pathlib import Path
ROOT=Path('/home/jhl/model_inference/llm_infer')
CODE=ROOT/'longbench_fixed_20260926'
OUT=ROOT/'longbench_outputs_qwen8b_fixed_20260926'
PYTHON='/home/jhl/anaconda3/envs/qwen-opt/bin/python'
TASKS=['qasper','hotpotqa','multifieldqa_en','2wikimqa','gov_report','qmsum','trec','repobench-p']
JOBS=[(tag,m,task) for task in TASKS for tag,m in [('full',80),('m64',64),('m80',80)]]
lock=threading.Lock()
state={'protocol':'nonthinking-eos-reserved-context-v1','started_at':time.strftime('%F %T'),'running':{},'completed':[],'failed':[]}
queue=iter(JOBS)

def save():
 p=OUT/'scheduler_state.json';tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(state,indent=2));tmp.replace(p)

def worker(gpu):
 while True:
  with lock:
   job=next(queue,None)
   if job is None:return
   tag,m,task=job;label=f'{tag}/{task}'
   state['running'][str(gpu)]=label;save()
  dest=OUT/tag/task
  cmd=[PYTHON,'-u',str(CODE/'run_qwen_longbench_fixed.py'),'--model-path','/data/share/Qwen3-8B','--data-dir',str(ROOT/'longbench_data'),'--prompt-config',str(ROOT/'dataset2prompt_longbench.json'),'--maxlen-config',str(ROOT/'dataset2maxlen_longbench.json'),'--runner-dir',str(ROOT),'--output-dir',str(dest),'--method','full' if tag=='full' else 'ours','--m',str(m),'--tasks',task,'--limit','0']
  env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',TOKENIZERS_PARALLELISM='false',PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True')
  logfile=OUT/'logs'/f'{tag}_{task}.log'
  with logfile.open('a') as f:
   f.write('\nSCHEDULER_START '+time.strftime('%F %T')+'\n');f.flush()
   result=subprocess.run(cmd,env=env,cwd=ROOT,stdout=f,stderr=subprocess.STDOUT)
   f.write('\nSCHEDULER_END '+time.strftime('%F %T')+f' rc={result.returncode}\n')
  with lock:
   state['running'].pop(str(gpu),None)
   state['completed' if result.returncode==0 else 'failed'].append({'job':label,'rc':result.returncode,'gpu':gpu,'log':str(logfile)});save()

if __name__=='__main__':
 (OUT/'logs').mkdir(parents=True,exist_ok=True)
 if (OUT/'scheduler_state.json').exists():raise RuntimeError('Existing scheduler state: inspect before restarting')
 with concurrent.futures.ThreadPoolExecutor(max_workers=6) as pool:
  list(pool.map(worker,range(6)))
 print('SCHEDULER_DONE',json.dumps(state),flush=True)
