from __future__ import annotations
import hashlib,importlib.util,os,sys,urllib.request
from pathlib import Path
RUNTIME_REPOSITORY='Anurag9000/RigorousRAG';RUNTIME_COMMIT='0c50adb23e5ba58b7c49b18401950f0bf7e5b736';RUNTIME_PATH='training/dataset_cohort_runtime.py';RUNTIME_BLOB='1ffb1af0f70812d3418a6f0959ae88c7a145939e';RUNTIME_URL=f'https://raw.githubusercontent.com/{RUNTIME_REPOSITORY}/{RUNTIME_COMMIT}/{RUNTIME_PATH}'
def git_blob_sha(data:bytes)->str:return hashlib.sha1(f'blob {len(data)}\0'.encode()+data).hexdigest()
def materialize_runtime(root:Path|None=None)->Path:
 r=(root or Path(os.environ.get('TRAINING_CONTROL_REPO_ROOT') or Path.cwd())).resolve();t=r/'.training_control'/'dataset_cohort'/RUNTIME_COMMIT/'dataset_cohort_runtime.py'
 if t.is_file() and git_blob_sha(t.read_bytes())==RUNTIME_BLOB:return t
 p=urllib.request.urlopen(urllib.request.Request(RUNTIME_URL,headers={'User-Agent':'opf-dataset-cohort-runtime/1'}),timeout=120).read();a=git_blob_sha(p)
 if a!=RUNTIME_BLOB:raise RuntimeError(f'Pinned dataset-cohort runtime checksum mismatch: {a} != {RUNTIME_BLOB}')
 t.parent.mkdir(parents=True,exist_ok=True);q=t.with_suffix('.py.tmp');q.write_bytes(p);os.replace(q,t);return t
def load_runtime(root:Path|None=None):
 p=materialize_runtime(root);n=f'_opf_dataset_cohort_runtime_{RUNTIME_COMMIT[:12]}'
 if n in sys.modules:return sys.modules[n]
 s=importlib.util.spec_from_file_location(n,p)
 if s is None or s.loader is None:raise RuntimeError(f'Cannot import dataset-cohort runtime: {p}')
 m=importlib.util.module_from_spec(s);sys.modules[n]=m;s.loader.exec_module(m);return m
