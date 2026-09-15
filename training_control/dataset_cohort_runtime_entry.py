"""Blob-verified loader for transactional dataset-cohort runtime v2."""
from __future__ import annotations
import hashlib,importlib.util,os,sys,urllib.request
from pathlib import Path
R='Anurag9000/RigorousRAG';C1='0c50adb23e5ba58b7c49b18401950f0bf7e5b736';P1='training/dataset_cohort_runtime.py';B1='1ffb1af0f70812d3418a6f0959ae88c7a145939e';C2='0124a824d4fff32f4b427812a27cf7937d7c0891';P2='training/dataset_cohort_runtime_v2.py';B2='9364847538404151966b3719b10a1a9dc5f0f400';RUNTIME_REPOSITORY=R;RUNTIME_COMMIT=C2;RUNTIME_PATH=P2;RUNTIME_BLOB=B2
def H(d):return hashlib.sha1(f'blob {len(d)}\0'.encode()+d).hexdigest()
def G(r,c,p,b):
 t=r/'.training_control'/'dataset_cohort'/c/Path(p).name
 if t.is_file() and H(t.read_bytes())==b:return t
 d=urllib.request.urlopen(urllib.request.Request(f'https://raw.githubusercontent.com/{R}/{c}/{p}',headers={'User-Agent':'opf-dataset-cohort-v2'}),timeout=120).read()
 if H(d)!=b:raise RuntimeError('cohort runtime checksum mismatch')
 t.parent.mkdir(parents=True,exist_ok=True);q=t.with_suffix(t.suffix+'.tmp');q.write_bytes(d);os.replace(q,t);return t
def I(n,p):
 if n in sys.modules:return sys.modules[n]
 s=importlib.util.spec_from_file_location(n,p);m=importlib.util.module_from_spec(s);sys.modules[n]=m;s.loader.exec_module(m);return m
def materialize_runtime(root=None):
 r=(root or Path(os.environ.get('TRAINING_CONTROL_REPO_ROOT') or '.')).resolve();G(r,C1,P1,B1);return G(r,C2,P2,B2)
def load_runtime(root=None):
 r=(root or Path(os.environ.get('TRAINING_CONTROL_REPO_ROOT') or '.')).resolve();I(f'_opf_dataset_cohort_runtime_{C1[:12]}',G(r,C1,P1,B1));m=I(f'_opf_dataset_cohort_runtime_v2_{C2[:12]}',G(r,C2,P2,B2));assert m.BASE_COMMIT==C1;return m
