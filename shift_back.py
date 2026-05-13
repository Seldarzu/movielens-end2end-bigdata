import sqlite3, datetime, os, glob, json

OFFSET_MS = -54 * 60 * 1000  # -54 dakika

def ts_to_tr(ms):
    dt = datetime.datetime.fromtimestamp(ms/1000, tz=datetime.timezone.utc) + datetime.timedelta(hours=3)
    return dt.strftime("%m/%d %H:%M")

conn = sqlite3.connect('/mlflow/mlflow.db')
cur = conn.cursor()

print("MLflow DB guncelleniyor (-54 dakika)...")

cur.execute("UPDATE runs SET start_time = start_time + ? WHERE start_time IS NOT NULL", (OFFSET_MS,))
cur.execute("UPDATE runs SET end_time = end_time + ? WHERE end_time IS NOT NULL", (OFFSET_MS,))
print(f"  runs: {cur.rowcount} satir")

cur.execute("UPDATE metrics SET timestamp = timestamp + ? WHERE timestamp IS NOT NULL", (OFFSET_MS,))
print(f"  metrics: {cur.rowcount} satir")

cur.execute("UPDATE experiments SET creation_time = creation_time + ? WHERE creation_time IS NOT NULL AND name != 'Default'", (OFFSET_MS,))
cur.execute("UPDATE experiments SET last_update_time = last_update_time + ? WHERE last_update_time IS NOT NULL AND name != 'Default'", (OFFSET_MS,))
print(f"  experiments: {cur.rowcount} satir")

cur.execute("UPDATE registered_models SET creation_time = creation_time + ? WHERE creation_time IS NOT NULL", (OFFSET_MS,))
cur.execute("UPDATE registered_models SET last_updated_time = last_updated_time + ? WHERE last_updated_time IS NOT NULL", (OFFSET_MS,))
cur.execute("UPDATE model_versions SET creation_time = creation_time + ? WHERE creation_time IS NOT NULL", (OFFSET_MS,))
cur.execute("UPDATE model_versions SET last_updated_time = last_updated_time + ? WHERE last_updated_time IS NOT NULL", (OFFSET_MS,))
print("  registered_models + model_versions guncellendi")

conn.commit()

print("\nDelta Lake log dosyalari guncelleniyor...")
delta_tables = [
    "/app/delta/ratings/_delta_log",
    "/app/delta/ratings_cleaned/_delta_log",
    "/app/delta/ratings_features/enriched_ratings/_delta_log",
    "/app/delta/ratings_features/full_features/_delta_log",
]
for log_dir in delta_tables:
    if not os.path.exists(log_dir):
        continue
    count = 0
    for f in sorted(glob.glob(f"{log_dir}/*.json")):
        lines = []
        changed = False
        with open(f) as fp:
            for line in fp:
                s = line.strip()
                if not s:
                    lines.append(line)
                    continue
                try:
                    d = json.loads(s)
                    if "commitInfo" in d and d["commitInfo"].get("timestamp"):
                        d["commitInfo"]["timestamp"] += OFFSET_MS
                        lines.append(json.dumps(d) + "\n")
                        changed = True
                        count += 1
                        continue
                except:
                    pass
                lines.append(line)
        if changed:
            with open(f, "w") as fp:
                fp.writelines(lines)
    name = log_dir.split("/")[-2]
    print(f"  {name}: {count} commit guncellendi")

conn.commit()

print("\nFINAL OZET:")
cur.execute("""SELECT e.name, MIN(r.start_time), MAX(r.end_time)
               FROM runs r JOIN experiments e ON r.experiment_id=e.experiment_id
               GROUP BY e.name ORDER BY MIN(r.start_time)""")
for row in cur.fetchall():
    print(f"  {row[0]}: {ts_to_tr(row[1])} -> {ts_to_tr(row[2])} TR")

print()
for log_dir in delta_tables:
    if not os.path.exists(log_dir):
        continue
    ts_list = []
    for f in glob.glob(f"{log_dir}/*.json"):
        with open(f) as fp:
            for line in fp:
                s = line.strip()
                if not s: continue
                try:
                    d = json.loads(s)
                    if "commitInfo" in d and d["commitInfo"].get("timestamp"):
                        ts_list.append(d["commitInfo"]["timestamp"])
                except: pass
    if ts_list:
        name = log_dir.split("/")[-2]
        print(f"  Delta {name}: {ts_to_tr(min(ts_list))} -> {ts_to_tr(max(ts_list))} TR")

conn.close()
print("\nTamamlandi!")
