#!/usr/bin/env python3
"""Compile actual lecture C++ functions, run sanitizers, and verify printed examples."""
import json, pathlib, subprocess, tempfile
root=pathlib.Path(__file__).resolve().parent
with tempfile.TemporaryDirectory(prefix='ece408-l11-') as tmp:
    exe=pathlib.Path(tmp)/'verify'
    subprocess.run(['clang++','-O1','-std=c++17','-Wall','-Wextra','-Werror','-pedantic','-fsanitize=undefined','-fno-omit-frame-pointer',str(root/'verify.cpp'),'-o',str(exe)],check=True)
    result=json.loads(subprocess.check_output([str(exe)],text=True))
X=[[[1,2,0,1],[1,1,3,2],[0,2,2,0],[2,1,0,3]],[[0,2,1,0],[0,3,2,1],[1,1,0,2],[2,1,0,3]],[[1,2,1,0],[0,1,3,2],[3,3,2,0],[1,3,2,0]]]
F=[[[1,1,1],[2,2,3],[2,1,0]],[[1,2,3],[1,1,0],[3,0,1]],[[0,1,1],[1,0,2],[1,2,1]]]
y=[[sum(X[c][h+p][w+q]*F[c][p][q] for c in range(3) for p in range(3) for q in range(3)) for w in range(2)] for h in range(2)]
assert y==[[51,50],[60,48]]
A=[[1,1,2,2,1,1,1,1,0,1,1,0],[1,0,0,1,2,1,2,1,1,2,2,0]]
small=[[[1,2,0],[1,1,3],[0,2,2]],[[0,2,1],[0,3,2],[1,1,0]],[[1,2,1],[0,1,3],[3,3,2]]]
U=[[small[c][h+p][w+q] for h in range(2) for w in range(2)] for c in range(3) for p in range(2) for q in range(2)]
V=[[sum(A[m][r]*U[r][l] for r in range(12)) for l in range(4)] for m in range(2)]
assert V==[[14,20,15,24],[12,24,17,26]]
assert 2*10*92*192*3*81==85847040
assert (20*20*22,16**3*175)==(8800,716800)
assert (20*165*22,16*161*16*175)==(72600,7212800)
assert 4*80**2*49==1254400 and 16*34**2*196==3625216
assert 4*49*6400*10000==12544000000
assert 4*196*1156*10000==9063040000
# Enumerate padded-window starts independently of the closed form.
assert list(range(0,6+2-3+1,2))==[0,2,4]
assert list(range(0,8+2-3+1,2))==[0,2,4,6]
result.update(printed_examples='PASS',stride_padding='PASS',cuda_compilation='NOT_RUN',gpu_execution='NOT_RUN',gpu_performance='NOT_RUN',address_sanitizer='BLOCKED_RUNTIME_INITIALIZATION',undefined_behavior_sanitizer='PASS')
(root/'verification.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2))
