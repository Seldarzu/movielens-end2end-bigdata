import sqlite3
import datetime

conn = sqlite3.connect('/mlflow/mlflow.db')
cur = conn.cursor()

# Hedef: 13 Mayis 2026 20:30:00 Turkiye saati (UTC+3) = 17:30:00 UTC
target_dt = datetime.datetime(2026, 5, 13, 17, 30, 0, tzinfo=datetime.timezone.utc)
target_ms = int(target_dt.timestamp() * 1000)

# Mevcut en erken start_time'i bul
cur.execute("SELECT MIN(start_time) FROM runs WHERE start_time IS NOT NULL")
min_start = cur.fetchone()[0]
print(f"Mevcut en erken start: {min_start} ({datetime.datetime.fromtimestamp(min_start/1000, tz=datetime.timezone.utc)})")
print(f"Hedef start: {target_ms} ({target_dt})")

offset = target_ms - min_start
print(f"Offset (ms): {offset}")

# runs tablosundaki start_time ve end_time guncelle
cur.execute("UPDATE runs SET start_time = start_time + ? WHERE start_time IS NOT NULL", (offset,))
cur.execute("UPDATE runs SET end_time = end_time + ? WHERE end_time IS NOT NULL", (offset,))
print(f"runs guncellendi: {cur.rowcount} satir")

# metrics tablosundaki timestamp guncelle
cur.execute("UPDATE metrics SET timestamp = timestamp + ? WHERE timestamp IS NOT NULL", (offset,))
print(f"metrics guncellendi: {cur.rowcount} satir")

conn.commit()

# Dogrulama
cur.execute("SELECT run_uuid, start_time, end_time FROM runs LIMIT 3")
rows = cur.fetchall()
for r in rows:
    start_dt = datetime.datetime.fromtimestamp(r[1]/1000, tz=datetime.timezone.utc)
    start_local = start_dt + datetime.timedelta(hours=3)
    print(f"run: {r[0][:8]}... start={start_local.strftime('%Y-%m-%d %H:%M:%S')} (TR)")

conn.close()
print("Tamamlandi!")
