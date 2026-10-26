import argparse
import json
from pathlib import Path
p=argparse.ArgumentParser(description="Initialize independent output directories and frozen REFED LOSO split")
p.add_argument("--work-dir",type=Path,default=Path("work"))
a=p.parse_args()
root=Path(__file__).resolve().parents[1]
for name in ["manifests","splits","audits","predictions","checkpoints","logs","features","cache"]:
    (a.work_dir/name).mkdir(parents=True,exist_ok=True)
target=a.work_dir/"splits"/"refed_strict_loso_v1.json"
value=json.loads((root/"configs"/"refed_strict_loso_v1.json").read_text(encoding="utf-8"))
if target.exists() and json.loads(target.read_text(encoding="utf-8"))!=value:
    raise ValueError("Existing split differs; refusing overwrite")
target.write_text(json.dumps(value,indent=2)+"\n",encoding="utf-8")
