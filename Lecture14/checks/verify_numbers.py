#!/usr/bin/env python3
"""Arithmetic checks only; no GPU benchmark or inherited code revalidation."""
import json,math,re,zipfile,hashlib
from pathlib import Path
r=Path(__file__).resolve().parents[1]
z=zipfile.ZipFile(r.parent/'Lec14_source.zip')
# Listings are unchanged apart from LaTeX presentation options.
listing_count=0
for p in (r/'sections').glob('*.tex'):
 old=z.read('Lec14/sections/'+p.name).decode();new=p.read_text()
 pattern=r'\\begin\{lstlisting\}(?:\[[^\n]*\])?\n(.*?)\\end\{lstlisting\}'
 a=re.findall(pattern,old,re.S);b=re.findall(pattern,new,re.S)
 assert a==b,p.name
 listing_count+=len(a)
for p in (r/'assets').glob('*.png'):
 assert p.read_bytes()==z.read('Lec14/assets/'+p.name)
sectors=[len({(stride*j+offset)//32 for j in range(32)}) for stride,offset in [(4,0),(4,4),(8,0),(32,0)]]
assert sectors==[4,5,8,32]
banks={str(s):max(sum((s*j)%32==b for j in range(32)) for b in range(32)) for s in [1,2,32,33]}
assert list(banks.values())==[1,2,32,1]
occ=lambda registers:min(16,1536//256,48//8,65536//(256*registers),100//16)*8/48
assert occ(32)==1 and occ(64)==2/3
peaks={'A40':37400,'L40S':65434,'B200':47309}
times={k:6.72/v*1000 for k,v in peaks.items()}
assert round(times['L40S'],4)==.1027
assert math.isclose((211.76-132.35)/211.76,.375)
assert round((225-145.59)/225*100,2)==35.29
assert 396/(2*264)==.75
report={'unchanged_listing_bodies':listing_count,'unchanged_screenshots':8,'sector_counts':sectors,'bank_conflict_degrees':banks,'occupancy_32_registers':occ(32),'occupancy_64_registers':occ(64),'compute_lower_bound_ms':times,'L40S_ideal_speedup':3.47/times['L40S'],'tail_slot_utilization':.75,'vector_load_reduction':.375,'shared_total_instruction_reduction_percent':(225-145.59)/225*100,'reported_benchmark_status':'source measurements, not rerun','cuda_compile':'not_run','gpu_execution':'not_run','gpu_profiling':'not_run'}
(r/'checks/numerical_results.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
