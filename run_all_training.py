#!/usr/bin/env python3
from __future__ import annotations
import hashlib,json,os,subprocess,sys,urllib.request
from pathlib import Path
R="Anurag9000/ESD_Project";C="45bfe868867512e7876b2623e641384686591f71";S="116c5e7c4694fb0f84e93d397a68f7cea15310e2";U=f"https://raw.githubusercontent.com/Anurag9000/RigorousRAG/{C}/tools/universal_training_controller_entry.py";D=Path(__file__).resolve().parent
P={"repository":R,"preferred_training_entrypoints":["train.py","main.py","run_training.py","scripts/train.py","scripts/run_training.py","scripts/run_pipeline.py"],"preferred_dataset_entrypoints":["prepare_data.py","scripts/prepare_data.py","scripts/download_data.py","scripts/dataset_setup.py"],"dynamic_registry_covers":[],"extra_jobs":[],"ignore_entrypoints":["run_all_training.py"],"strict_coverage":True,"require_native_resume":True,"require_exact_resume":True,"require_training_exact_resume":True,"require_training_early_stopping":True,"require_dag_enforcement":True,"require_model_surface_accounting":True,"require_literal_opf_mechanism_parity":True,"require_well_formed_training_exemptions":True}
def h(x):return hashlib.sha1(f"blob {len(x)}\0".encode()+x).hexdigest()
def main():
 q=D/".training_control"/"universal_training_controller_entry.py"
 if not q.is_file() or h(q.read_bytes())!=S:
  q.parent.mkdir(parents=True,exist_ok=True);x=urllib.request.urlopen(U,timeout=60).read()
  if h(x)!=S:raise RuntimeError("Pinned controller checksum mismatch")
  t=q.with_suffix(".tmp");t.write_bytes(x);os.replace(t,q)
 e=os.environ.copy();e["TRAINING_CONTROL_PROFILE"]=json.dumps(P,separators=(",",":"));e["TRAINING_CONTROL_REPO_ROOT"]=str(D);e.setdefault("TRAINING_CONTROL_TERMINATION_GRACE_SEC","30");return subprocess.call([sys.executable,str(q),*sys.argv[1:]],cwd=D,env=e)
if __name__=="__main__":raise SystemExit(main())
