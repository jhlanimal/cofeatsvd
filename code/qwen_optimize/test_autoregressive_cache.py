import sys
from pathlib import Path
import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from autoregressive_cache import LayerCache
from eval_qwen3_multiseq_ppl_diagnostics import weighted_coalescing_page

def main():
    torch.manual_seed(0); p,d=32,128; scale=d**-.5
    kpage=torch.randn(p,d); qpage=torch.randn(p,d); vpage=torch.randn(p,d)
    ref_c,ref_a=weighted_coalescing_page(kpage,qpage,64)
    full=LayerCache(); ours=LayerCache(method='ours',m=64)
    full_out=[]; ours_out=[]
    for t in range(96):
        q=torch.randn(d); k=torch.randn(d); v=torch.randn(d)
        if t < p: q,k,v=qpage[t],kpage[t],vpage[t]
        full_out.append(full.append_and_attend(q,k,v,scale)); ours_out.append(ours.append_and_attend(q,k,v,scale))
    sealed_c,sealed_a=ours.sealed_k[0]
    kh_ref=ref_c@ref_a.T; kh_stream=sealed_c@sealed_a.T
    rec_rel=float(torch.linalg.norm(kh_ref-kh_stream)/torch.linalg.norm(kh_ref))
    future=torch.randn(11,d); s1=future@kh_ref.T; s2=future@kh_stream.T
    score_rel=float(torch.linalg.norm(s1-s2)/torch.linalg.norm(s1)); score_max=float((s1-s2).abs().max())
    boundary=[]
    for t in (31,32,33,63,64,65,95,96): boundary.append({'token':t,'active_len':t%32,'sealed_pages':t//32})
    assert full.sealing_event_count==ours.sealing_event_count==3
    assert ours.compressed_history_calls>0 and not ours.full_history_fallback
    assert score_max <= 1e-4, (rec_rel,score_rel,score_max)
    print({'page_reconstruction_rel':rec_rel,'future_score_rel':score_rel,'future_score_max_abs':score_max,'boundaries':boundary,'full_seals':full.sealing_event_count,'ours_seals':ours.sealing_event_count,'compressed_history_calls':ours.compressed_history_calls})
if __name__=='__main__': main()
