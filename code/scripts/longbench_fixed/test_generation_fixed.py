import unittest
import torch
from generation_fixed import decode_loop, build_prompt, stop_ids
from types import SimpleNamespace

class GenerationTests(unittest.TestCase):
    def test_immediate_stop(self):
        def never(*args):raise AssertionError('Must not feed stop token')
        ids,reason=decode_loop(torch.tensor([9]),never,20,4,{9})
        self.assertEqual((ids,reason),([9],'eos'))
    def test_stop_during_decode(self):
        calls=[]
        def step(token,pos):
            calls.append((token.item(),pos))
            logits=torch.zeros(1,10);logits[0,9]=1;return logits
        ids,reason=decode_loop(torch.tensor([3]),step,20,4,{9})
        self.assertEqual((ids,reason),([3,9],'eos'))
        self.assertEqual(calls,[(3,20)])
    def test_budget_no_extra_forward(self):
        calls=[]
        def step(token,pos):
            calls.append(pos);return torch.tensor([[0.,1.]])
        ids,reason=decode_loop(torch.tensor([1]),step,20,3,{9})
        self.assertEqual((ids,reason),([1,1,1],'max_new_tokens'))
        self.assertEqual(calls,[20,21])
    def test_stop_union(self):
        model=SimpleNamespace(generation_config=SimpleNamespace(eos_token_id=[4,5]))
        tok=SimpleNamespace(eos_token_id=6)
        self.assertEqual(stop_ids(model,tok),{4,5,6})
    def test_impossible_prompt_budget(self):
        with self.assertRaises(ValueError):build_prompt(None,'x',True,10,10)

if __name__=='__main__':unittest.main()
