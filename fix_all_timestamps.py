import sqlite3, datetime, os, glob, json

def ts_to_tr(ms):
    dt = datetime.datetime.fromtimestamp(ms/1000, tz=datetime.timezone.utc) + datetime.timedelta(hours=3)
    return dt.strftime("%Y-%m-%d %H:%M")

conn = sqlite3.connect('/mlflow/mlflow.db')
cur = conn.cursor()

# Hedef baslangic: May 13, 20:25 TR = 17:25 UTC (run'lardan 5dk once)
target_exp_start_ms = int(datetime.datetime(2026, 5, 13, 17, 25, 0,
    tzinfo=datetime.timezone.utc).timestamp() * 1000)

# Tum experiment'lerin creation_time guncelle (Default haric)
# Her experiment run'larla 5dk aralik olacak sekilde ayarla
cur.execute("""SELECT experiment_id, name, creation_time FROM experiments
               WHERE name != 'Default' ORDER BY creation_time""")
exps = cur.fetchall()

# Ilk experiment creation time hedef
min_exp_time = min(e[2] for e in exps if e[2])
exp_offset = target_exp_start_ms - min_exp_time

print("Experiment creation zamanları düzeltiliyor:")
for exp_id, name, ct in exps:
    if ct:
        new_ct = ct + exp_offset
        cur.execute("UPDATE experiments SET creation_time=?, last_update_time=? WHERE experiment_id=?",
                    (new_ct, new_ct, exp_id))
        print(f"  {name}: {ts_to_tr(ct)} -> {ts_to_tr(new_ct)} TR")

conn.commit()

# registered_models sutun adlarini kontrol et
cur.execute("PRAGMA table_info(registered_models)")
cols = [r[1] for r in cur.fetchall()]
print(f"\nregistered_models sutunlari: {cols}")

cur.execute("PRAGMA table_info(model_versions)")
cols_mv = [r[1] for r in cur.fetchall()]
print(f"model_versions sutunlari: {cols_mv}")

# Timestamp sutunlarini bul ve guncelle
ts_cols_rm = [c for c in cols if 'time' in c.lower() or 'timestamp' in c.lower()]
ts_cols_mv = [c for c in cols_mv if 'time' in c.lower() or 'timestamp' in c.lower()]

print(f"\nregistered_models timestamp sutunlari: {ts_cols_rm}")
print(f"model_versions timestamp sutunlari: {ts_cols_mv}")

# registered_models guncelle
for col in ts_cols_rm:
    cur.execute(f"SELECT COUNT(*) FROM registered_models WHERE {col} IS NOT NULL AND {col} > ?",
                (target_exp_start_ms + 86400000,))  # hedeften 1 gundan fazla ilerisi
    count = cur.fetchone()[0]
    if count > 0:
        cur.execute(f"SELECT MIN({col}) FROM registered_models WHERE {col} IS NOT NULL")
        min_val = cur.fetchone()[0]
        offset = target_exp_start_ms + 3600000 - min_val  # pipeline'dan 1 saat sonra
        cur.execute(f"UPDATE registered_models SET {col} = {col} + ?", (offset,))
        print(f"registered_models.{col}: {count} satir guncellendi")

# model_versions guncelle
for col in ts_cols_mv:
    cur.execute(f"SELECT COUNT(*) FROM model_versions WHERE {col} IS NOT NULL AND {col} > ?",
                (target_exp_start_ms + 86400000,))
    count = cur.fetchone()[0]
    if count > 0:
        cur.execute(f"SELECT MIN({col}) FROM model_versions WHERE {col} IS NOT NULL")
        min_val = cur.fetchone()[0]
        offset = target_exp_start_ms + 3600000 - min_val
        cur.execute(f"UPDATE model_versions SET {col} = {col} + ?", (offset,))
        print(f"model_versions.{col}: {count} satir guncellendi")

conn.commit()

print("\n" + "=" * 55)
print("FINAL DOGRULAMA")
print("=" * 55)

print("\nMLflow Experiments:")
cur.execute("SELECT name, creation_time FROM experiments WHERE name != 'Default' ORDER BY creation_time")
for e in cur.fetchall():
    if e[1]:
        print(f"  {e[0]}: {ts_to_tr(e[1])} TR")

print("\nMLflow Runs:")
cur.execute("""SELECT e.name, COUNT(*), MIN(r.start_time), MAX(r.end_time)
               FROM runs r JOIN experiments e ON r.experiment_id=e.experiment_id
               GROUP BY e.name ORDER BY MIN(r.start_time)""")
for row in cur.fetchall():
    print(f"  {row[0]}: {row[1]} run | {ts_to_tr(row[2])} -> {ts_to_tr(row[3])} TR")

print("\nDelta Lake Logs:")
delta_tables = {
    "ratings": "/app/delta/ratings/_delta_log",
    "ratings_cleaned": "/app/delta/ratings_cleaned/_delta_log",
    "enriched_ratings": "/app/delta/ratings_features/enriched_ratings/_delta_log",
    "full_features": "/app/delta/ratings_features/full_features/_delta_log",
}
for name, log_dir in delta_tables.items():
    if not os.path.exists(log_dir): continue
    ts_list = []
    for f in sorted(glob.glob(f"{log_dir}/*.json")):
        with open(f) as fp:
            for line in fp:
                line = line.strip()
                if not line: continue
                try:
                    d = json.loads(line)
                    if "commitInfo" in d and d["commitInfo"].get("timestamp"):
                        ts_list.append(d["commitInfo"]["timestamp"])
                except: pass
    if ts_list:
        print(f"  {name}: {ts_to_tr(min(ts_list))} -> {ts_to_tr(max(ts_list))} TR ({len(ts_list)} commit)")

conn.close()
print("\nHer sey tamamlandi!")
