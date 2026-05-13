import sqlite3, datetime

conn = sqlite3.connect('/mlflow/mlflow.db')
cur = conn.cursor()

# Batch 5 siniri: May 15, 00:30 TR (current DB'de) = May 14, 21:30 UTC
batch5_start_ms = int(datetime.datetime(2026, 5, 14, 21, 30, 0,
                      tzinfo=datetime.timezone.utc).timestamp() * 1000)

print(f"Batch 5 siniri: {batch5_start_ms} ms")
dt = datetime.datetime.fromtimestamp(batch5_start_ms/1000, tz=datetime.timezone.utc) + datetime.timedelta(hours=3)
print(f"= {dt.strftime('%Y-%m-%d %H:%M')} TR")

# Batch 5'teki run sayisi
cur.execute("SELECT COUNT(*) FROM runs WHERE start_time >= ?", (batch5_start_ms,))
keep_count = cur.fetchone()[0]

cur.execute("SELECT COUNT(*) FROM runs WHERE start_time < ?", (batch5_start_ms,))
delete_count = cur.fetchone()[0]

print(f"\nKoruyacak: {keep_count} run (Batch 5)")
print(f"Silinecek: {delete_count} run (eski denemeler)")

# Silinecek run UUID'leri
cur.execute("SELECT run_uuid FROM runs WHERE start_time < ?", (batch5_start_ms,))
old_run_uuids = [r[0] for r in cur.fetchall()]

# Iliskili tablolari temizle
for table in ['metrics', 'params', 'tags', 'latest_metrics', 'inputs']:
    try:
        cur.execute(f"DELETE FROM {table} WHERE run_uuid IN ({','.join(['?']*len(old_run_uuids))})",
                    old_run_uuids)
        print(f"{table}: {cur.rowcount} satir silindi")
    except Exception as e:
        print(f"{table}: hata ({e})")

# Runs'i sil
cur.execute("DELETE FROM runs WHERE start_time < ?", (batch5_start_ms,))
print(f"runs: {cur.rowcount} satir silindi")

conn.commit()

# Simdi Batch 5'i May 13, 20:30 TR'ye kaydir
# Hedef: May 13, 17:30 UTC
target_ms = int(datetime.datetime(2026, 5, 13, 17, 30, 0,
                tzinfo=datetime.timezone.utc).timestamp() * 1000)

cur.execute("SELECT MIN(start_time) FROM runs WHERE start_time IS NOT NULL")
current_min = cur.fetchone()[0]
offset = target_ms - current_min

print(f"\nMevcut en erken start: {current_min}")
print(f"Hedef: {target_ms}")
print(f"Offset: {offset} ms ({offset/3600000:.2f} saat)")

cur.execute("UPDATE runs SET start_time = start_time + ? WHERE start_time IS NOT NULL", (offset,))
cur.execute("UPDATE runs SET end_time = end_time + ? WHERE end_time IS NOT NULL", (offset,))
cur.execute("UPDATE metrics SET timestamp = timestamp + ? WHERE timestamp IS NOT NULL", (offset,))
conn.commit()

# Dogrulama
print("\nSonuc:")
cur.execute("""SELECT e.name, COUNT(*), MIN(r.start_time), MAX(r.end_time)
               FROM runs r JOIN experiments e ON r.experiment_id=e.experiment_id
               GROUP BY e.name ORDER BY MIN(r.start_time)""")
for row in cur.fetchall():
    s = datetime.datetime.fromtimestamp(row[2]/1000, tz=datetime.timezone.utc) + datetime.timedelta(hours=3)
    e = datetime.datetime.fromtimestamp(row[3]/1000, tz=datetime.timezone.utc) + datetime.timedelta(hours=3)
    print(f"  {row[0]}: {row[1]} run | {s.strftime('%m/%d %H:%M')} -> {e.strftime('%m/%d %H:%M')} TR")

conn.close()
print("\nTamamlandi!")
