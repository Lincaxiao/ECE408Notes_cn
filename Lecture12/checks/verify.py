#!/usr/bin/env python3
"""CPU references and a float32 model of CUDA thread ownership and reductions."""
import json
from pathlib import Path
import numpy as np
rng=np.random.default_rng(40812)
def softmax(s):
    e=np.exp(s-np.max(s));return e/e.sum()
def tree(values,maximum):
    a=values.copy();step=len(a)//2
    while step:
        a[:step]=np.maximum(a[:step],a[step:2*step]) if maximum else a[:step]+a[step:2*step]
        step//=2
    return a[0]
def kernel_cpu(s):
    n=len(s);p=np.empty_like(s);bt=256
    for i in range(n):
        mx=np.full(bt,-np.inf,dtype=np.float32)
        for t in range(bt):
            for j in range(t,i+1,bt):mx[t]=max(mx[t],s[i,j])
        m=tree(mx,True);sm=np.zeros(bt,dtype=np.float32)
        for t in range(bt):
            for j in range(t,i+1,bt):
                p[i,j]=np.exp(np.float32(s[i,j]-m));sm[t]+=p[i,j]
        total=tree(sm,False)
        for t in range(bt):
            for j in range(t,n,bt):p[i,j]=p[i,j]/total if j<=i else 0
    return p
errors=[]
for n in [1,2,255,256,257,513]:
    s=(rng.normal(size=(n,n))*30+1000).astype(np.float32)
    upper=np.triu_indices(n,1);s[upper]=np.resize([np.nan,np.inf,-np.inf],len(upper[0]))
    p=kernel_cpu(s);ref=np.zeros((n,n))
    for i in range(n):ref[i,:i+1]=softmax(s[i,:i+1].astype(float))
    err=float(np.max(abs(p-ref)));errors.append({'N':n,'max_abs_error':err})
    assert err<2e-6 and np.isfinite(p).all() and (p[upper]==0).all()
    assert np.max(abs(p.sum(axis=1)-1))<2e-6
q=np.array([[2**.5,0],[0,2**.5],[2**.5,2**.5]])
k=np.array([[1.,0],[0,1],[1,1]]);v=np.array([[1.,0],[0,2],[3,1]])
s=q@k.T/2**.5;p=np.zeros((3,3))
for i in range(3):p[i,:i+1]=softmax(s[i,:i+1])
o=p@v
assert np.allclose(o,[[1,0],[.26894142137,1.46211715726],[1.94029221191,1]],atol=1e-9)
# Full-prefix vs cached new rows for arbitrary projections and positions.
x=rng.normal(size=(19,8));w=[rng.normal(size=(8,4)) for _ in range(3)]
Q,K,V=[x@a for a in w];cache_error=0.
for n in range(1,20):
    full=softmax(Q[n-1]@K[:n].T/2)@V[:n]
    ck=np.concatenate([K[:n-1],x[n-1:n]@w[1]])
    cv=np.concatenate([V[:n-1],x[n-1:n]@w[2]])
    inc=softmax((x[n-1]@w[0])@ck.T/2)@cv
    cache_error=max(cache_error,float(np.max(abs(full-inc))))
assert cache_error<1e-12
# Online recurrence: late maximum, partial final tile, entirely masked tiles.
online_error=0.;s=np.linspace(-100,100,37);v=rng.normal(size=(37,5))
for visible in [1,7,19,37]:
 for tile in [1,3,8,16,64]:
    m=None;den=0.;u=np.zeros(5)
    for start in range(0,37,tile):
        end=min(start+tile,visible)
        if end<=start:continue
        sj=s[start:end];vj=v[start:end];new=max(float(np.max(sj)),m if m is not None else -np.inf)
        alpha=np.exp(m-new) if m is not None else 0.
        e=np.exp(sj-new);den=alpha*den+sum(e);u=alpha*u+e@vj;m=new
    err=float(np.max(abs(u/den-softmax(s[:visible])@v[:visible])))
    online_error=max(online_error,err)
assert online_error<1e-12
params=4*96*12288**2;kv=2*1*32*4096*4096*2
ai=lambda B:2*B*1024*4096/(2*(1024*4096+B*1024+B*4096))
assert params==57982058496 and params*2/2**30==108 and kv/2**30==2
assert kv*8/32/2**30==.5
lengths=[5,11,17];variable=2*32*32*128*2*sum(lengths)
assert variable==sum(2*32*4096*2*n for n in lengths)
report={'softmax_cases':errors,'attention_example':o.tolist(),'cached_attention_max_error':cache_error,'online_softmax_max_error':online_error,'projection_parameters':params,'projection_GiB':108,'projection_GFLOP':2*params/1e9,'MHA_cache_GiB':2,'GQA_cache_GiB':.5,'variable_lengths':lengths,'variable_cache_bytes':variable,'batch_AI':{'1':ai(1),'512':ai(512)},'cuda_compile':'not_run','gpu_execution':'not_run','gpu_timing':'not_run'}
Path(__file__).with_name('results.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2))
