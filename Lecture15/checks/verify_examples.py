#!/usr/bin/env python3
"""Check prose arithmetic and source preservation; not a CUDA correctness test."""
from pathlib import Path
import json, re, zipfile
r=Path(__file__).resolve().parents[1]
z=zipfile.ZipFile(r.parent/'Lec15_source.zip')
assert (r/'examples/reduction_examples.cuh').read_bytes()==z.read('Lec15/examples/reduction_examples.cuh')
blocks=0
for p in (r/'sections').glob('*.tex'):
    old=z.read('Lec15/sections/'+p.name).decode()
    a=re.findall(r'\\begin\{cudacode\}\{[^\n]+\}\n(.*?)\\end\{cudacode\}',old,re.S)
    b=re.findall(r'\\begin\{lstlisting\}[^\n]*\n(.*?)\\end\{lstlisting\}',p.read_text(),re.S)
    assert a==b,p.name
    blocks+=len(a)
assert sum([4,7,2,3,8,5,9,6])==44
inc=[];dec=[]
for k in range(8):
    s=2**k
    inc.append(len({t//32 for t in range(128) if t%s==0}))
    dec.append(len({t//32 for t in range(128) if t<128//s}))
assert inc==[4,4,4,4,4,4,2,1] and dec==[4,2,1,1,1,1,1,1]
assert 32*sum(inc)==864 and 32*sum(dec)==384
assert 2*(4-1)+8==14 and 4*8==32
assert 1000000==488*2048+576
assert (1000000+2047)//2048==489
assert (489+2047)//2048==1
assert 512+488==1000
values=[sum(i<1000 for i in [512+t,640+t,768+t,896+t]) for t in range(128)]
assert values==[4]*104+[3]*24
assert [sum(values[i:i+32]) for i in range(0,128,32)]==[128,128,128,104]
# Index formula and boundary claims in prose, across incomplete tiles.
cases=0
for B in [32,96,128,256]:
 for C in [1,2,4]:
  M=2*B*C
  for n in [0,1,31,32,M-1,M,M+1,1000]:
   visits=[0]*n
   for block in range((n+M-1)//M):
    for t in range(B):
     for j in range(2*C):
      i=M*block+t+j*B
      if i<n: visits[i]+=1
   assert visits==[1]*n
   cases+=1
out={'prose_index_cases':cases,'numeric_examples':'passed','warp_utilization':[255/864,255/384],
'unchanged_listing_blocks':blocks,'implementation_file':'byte-identical to source archive',
'cuda_correctness_retesting':'not performed; code unchanged from imported source',
'cuda_compile':'not run','gpu_execution':'not run','gpu_timing':'not run'}
(r/'checks/verification.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
