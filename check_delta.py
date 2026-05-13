import json, os, glob

log_dir = "/app/delta/ratings/_delta_log"
files = sorted(glob.glob(f"{log_dir}/*.json"))[:3]
for f in files:
    with open(f) as fp:
        for line in fp:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            if "commitInfo" in d:
                ci = d["commitInfo"]
                print(f"{os.path.basename(f)}: timestamp={ci.get('timestamp')}, operation={ci.get('operation')}")
                break
