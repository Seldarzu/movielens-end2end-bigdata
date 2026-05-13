import sqlite3, datetime

conn = sqlite3.connect('/mlflow/mlflow.db')
cur = conn.cursor()

# En uzun sureleri goster
cur.execute("""SELECT r.run_uuid, e.name, r.start_time, r.end_time,
               (r.end_time - r.start_time) as duration_ms
               FROM runs r JOIN experiments e ON r.experiment_id=e.experiment_id
               ORDER BY duration_ms DESC LIMIT 10""")

for row in cur.fetchall():
    start_dt = datetime.datetime.fromtimestamp(row[2]/1000, tz=datetime.timezone.utc) + datetime.timedelta(hours=3)
    end_dt = datetime.datetime.fromtimestamp(row[3]/1000, tz=datetime.timezone.utc) + datetime.timedelta(hours=3)
    dur_sec = row[4] / 1000
    print(f"{row[1][:30]}: {start_dt.strftime('%m/%d %H:%M')} -> {end_dt.strftime('%m/%d %H:%M')} ({dur_sec:.0f}s)")

conn.close()
