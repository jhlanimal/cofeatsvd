"""Periodically score snapshots; produce final scores once scheduler ends."""
import json,subprocess,time
from pathlib import Path
ROOT=Path('/home/jhl/model_inference/llm_infer/longbench_outputs_qwen8b_fixed_20260926')
CMD=['/home/jhl/anaconda3/envs/qwen-opt/bin/python','/home/jhl/model_inference/llm_infer/longbench_fixed_20260926/score_fixed.py']
while True:
 p=ROOT/'scheduler_state.json'
 if p.exists():
  try:state=json.loads(p.read_text())
  except json.JSONDecodeError:time.sleep(10);continue
  result=subprocess.run(CMD,capture_output=True,text=True)
  print(time.strftime('%F %T'),'score_rc',result.returncode,'completed',len(state['completed']),'failed',len(state['failed']),flush=True)
  if result.returncode:print(result.stderr,flush=True)
  if len(state['completed'])+len(state['failed'])==24 and not state['running']:break
 time.sleep(180)
