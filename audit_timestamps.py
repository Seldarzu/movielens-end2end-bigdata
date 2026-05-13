import os, json, glob, datetime

def ts_to_tr(ms):
    if ms is None:
        return "?"
    dt = datetime.datetime.fromtimestamp(ms/1000, tz=datetime.timezone.utc) + datetime.timedelta(hours=3)
    return dt.strftime("%Y-%m-%d %H:%M")

print("=" * 60)
print("DELTA LAKE LOG TIMESTAMPS")
print("=" * 60)

delta_roots = [
    "/app/delta/ratings",
    "/app/delta/ratings_cleaned",
    "/app/delta/ratings_features/enriched_ratings",
    "/app/delta/ratings_features/full_features",
]

for root in delta_roots:
    log_dir = f"{root}/_delta_log"
    if not os.path.exists(log_dir):
        print(f"{root}: yok")
        continue
    files = sorted(glob.glob(f"{log_dir}/*.json"))
    if not files:
        print(f"{root}: json yok")
        continue
    ts_list = []
    for f in files:
        with open(f) as fp:
            for line in fp:
                line = line.strip()
                if not line:
                    continue
                try:
                    d = json.loads(line)
                    if "commitInfo" in d:
                        ts = d["commitInfo"].get("timestamp")
                        if ts:
                            ts_list.append(ts)
                except:
                    pass
    if ts_list:
        print(f"{root.split('/')[-1]}: {len(ts_list)} commit | {ts_to_tr(min(ts_list))} -> {ts_to_tr(max(ts_list))} TR")
    else:
        print(f"{root}: timestamp bulunamadi")

print()
print("=" * 60)
print("EDA / OUTPUT DOSYALARI")
print("=" * 60)

for path in ["/app/delta/eda", "/app/delta/scale_results"]:
    if os.path.exists(path):
        files = glob.glob(f"{path}/**/*", recursive=True)
        print(f"{path}: {len(files)} dosya var")
    else:
        print(f"{path}: yok")

print()
print("=" * 60)
print("MLFLOW ARTIFACTS")
print("=" * 60)

artifact_root = "/mlflow/artifacts"
if os.path.exists(artifact_root):
    all_files = glob.glob(f"{artifact_root}/**/*", recursive=True)
    print(f"Toplam artifact: {len(all_files)} dosya")
    exts = {}
    for f in all_files:
        if os.path.isfile(f):
            ext = os.path.splitext(f)[1] or "(no ext)"
            exts[ext] = exts.get(ext, 0) + 1
    for ext, cnt in sorted(exts.items()):
        print(f"  {ext}: {cnt}")
else:
    print("Artifact dizini yok")

print()
print("=" * 60)
print("SPARK CIKTI DOSYALARI (grafik/png)")
print("=" * 60)
png_files = glob.glob("/app/spark/**/*.png", recursive=True) + glob.glob("/app/delta/**/*.png", recursive=True)
print(f"PNG dosyasi: {len(png_files)}")
