"""Relaunch ONLY Arm C (A and B results already captured). Writes a separate results_c.json."""
import sys, os, time, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.stdout.reconfigure(line_buffering=True)
import torch
import experiments.edt_ab.ablib as ablib
torch.set_num_threads(os.cpu_count() or 8)
print('loading split...', flush=True)
split = ablib.load_corpus('data/communication_corpus.pt', n_train=400000, n_holdout=30000, n_phase1=int(400000*0.15))
print(f"domain_split sizes: {[d.numel() for d in split['domain_split']]}", flush=True)
print('building engine...', flush=True)
eng = ablib.build_engine(seed=42)
print('=== Arm C_edt_spec ===', flush=True)
t0 = time.time()
out = ablib.arm_edt_spec(eng, train=split['train'], holdout=split['holdout'], budget=400000, domain_split=split['domain_split'], chunk_len=16, lr=3e-4)
out['wall_clock_s'] = time.time() - t0
with open('experiments/edt_ab/results_c.json','w') as f:
    summary = {k:v for k,v in out.items() if k not in ('losses','phase1_losses','phase2b_losses','phase3_losses')}
    summary['phase3_final_loss'] = (out.get('phase3_losses') or [0])[-1]
    json.dump(summary, f, indent=2)
with open('experiments/edt_ab/samples/C_edt_spec.txt','w') as f:
    f.write(out.get('sample',''))
print(f"Arm C DONE: ppl={out['ppl']:.2f} div={out['diversity']:.3f} acc={out.get('accuracy')} time={out['wall_clock_s']:.0f}s", flush=True)
print('wrote experiments/edt_ab/results_c.json', flush=True)
