#!/usr/bin/env python3
"""CPU index/count model; does not simulate CUDA memory races or timing."""
from collections import Counter
from pathlib import Path
import json, random, re

def simulate(xs, blocks, threads):
    visits = [0] * len(xs)
    merged = Counter()
    aggregated = Counter()
    global_ops = shared_ops = runs = 0
    for block in range(blocks):
        local = Counter()
        local_agg = Counter()
        init = [0] * 256
        commit = [0] * 256
        for thread in range(threads):
            for b in range(thread, 256, threads):
                init[b] += 1
                commit[b] += 1
            last = None
            count = 0
            for i in range(block * threads + thread, len(xs), blocks * threads):
                visits[i] += 1
                b = xs[i]
                local[b] += 1
                shared_ops += 1
                if count and b == last:
                    count += 1
                else:
                    if count:
                        local_agg[last] += count
                        runs += 1
                    last, count = b, 1
            if count:
                local_agg[last] += count
                runs += 1
        assert init == commit == [1] * 256
        assert local == local_agg
        global_ops += len(local)
        merged.update(local)
        aggregated.update(local_agg)
    assert visits == [1] * len(xs)
    assert merged == aggregated == Counter(xs)
    assert shared_ops == len(xs)
    assert runs <= len(xs)
    assert global_ops <= min(len(xs), blocks * 256)
    return merged, global_ops, runs

def main():
    rng = random.Random(40813)
    cases = 0
    for n in [0, 1, 14, 16, 31, 257, 1031]:
        inputs = [[7]*n, [i%256 for i in range(n)], [i%2 for i in range(n)],
                  [rng.randrange(256) for _ in range(n)]]
        for xs in inputs:
            for blocks in [1, 2, 7]:
                for threads in [1, 32, 128, 256, 512]:
                    simulate(xs, blocks, threads)
                    cases += 1
    xs = [0,2,1,2,3,2,0,1,2,3,3,1,0,2,3,2]
    full, ops, _ = simulate(xs,2,4)
    short, _, _ = simulate(xs[:14],2,4)
    assert [full[i] for i in range(4)] == [3,3,6,4] and ops == 8
    assert [short[i] for i in range(4)] == [3,3,5,3]
    assert simulate([7,7,2,2,2,7],1,1)[2] == 3
    assert simulate([0,0,1,1,0,0,1,1],1,2)[2] == 8
    assert simulate([0,1,0,1,0,1,0,1],1,2)[2] == 2
    assert [x//64 for x in [0,63,64,255]] == [0,0,1,3]
    letters=Counter(c.upper() for c in 'Programming Massively Parallel Processors' if c.isalpha())
    assert letters == dict(A=4,C=1,E=3,G=2,I=2,L=4,M=3,N=1,O=3,P=3,R=5,S=5,V=1,Y=1)
    assert sum(letters.values()) == 38
    p = (700-100/0.2424)/575
    assert abs(p-0.5)<1e-4
    coarse = {c: (2**20+c*256-1)//(c*256) for c in [1,8]}
    assert coarse == {1:4096,8:512}
    root=Path(__file__).resolve().parents[1]
    code=(root/'code/histogram.cu').read_text()
    listings=[]
    for name in ['01_atomic.tex','02_histogram.tex','04_privatization.tex']:
        listings.extend(re.findall(r'\\begin\{lstlisting\}[^\n]*\n(.*?)\\end\{lstlisting\}',(root/'sections'/name).read_text(),re.S))
    for listing in listings:
        assert listing.strip() in code
    result={'cpu_cases':cases,'index_coverage':'passed','per_bin_counts':'passed',
            'aggregation_and_final_flush':'passed','numeric_examples':'passed',
            'listing_code_consistency':len(listings),'l2_fraction':p,
            'cuda_compile':'not run','gpu_execution':'not run','gpu_timing':'not run'}
    out=root/'checks/verification.json';out.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))
if __name__=='__main__': main()
