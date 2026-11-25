"""Exercise the separately launched worker over HTTP and reconnect with a fresh client."""
import csv
import io
import os
import socket
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import httpx


def main():
    with tempfile.TemporaryDirectory() as home:
        with socket.socket() as sock:
            sock.bind(("127.0.0.1",0))
            port=sock.getsockname()[1]
        env={**os.environ,"QSENSE_HOME":home}
        base=f"http://127.0.0.1:{port}"
        server=subprocess.Popen([sys.executable,"start.py","--no-browser","--port",str(port)],env=env)
        try:
            for _ in range(150):
                try:
                    if httpx.get(base,timeout=2).status_code==200:
                        break
                except httpx.TransportError:
                    time.sleep(.2)
            with httpx.Client(base_url=base,headers={"x-qsense":"studio"},timeout=60) as client:
                file=io.StringIO()
                writer=csv.writer(file)
                writer.writerow(["date","rfh","version"])
                for year in range(2000,2012):
                    for month in range(1,13):
                        for day in (1,11,21):
                            writer.writerow([f"{year}-{month:02}-{day:02}",month+day*.7,"final"])
                r=client.post("/api/stage",files={"files":("rain.csv",file.getvalue(),"text/csv")})
                r.raise_for_status()
                source=r.json()[0]["id"]
                client.post(f"/api/stage/{source}/commit").raise_for_status()
                config={"dataset":source,"hidden":4,"epochs":14,"arms":["qt"],"lookback":4}
                r=client.post("/api/runs",json=config)
                r.raise_for_status()
                key=r.json()["id"]
                url="/api/runs/"+key
                for _ in range(600):
                    job=client.get(url).json()
                    if job["state"]["status"]=="failed":
                        raise RuntimeError(job["state"])
                    if job["progress"].get("history"):
                        break
                    time.sleep(.04)
                client.post(url+"/pause").raise_for_status()
                for _ in range(200):
                    if client.get(url).json()["state"]["status"]=="paused":
                        break
                    time.sleep(.1)
                client.post(url+"/resume").raise_for_status()
            with httpx.Client(base_url=base,timeout=30) as fresh:
                for _ in range(600):
                    job=fresh.get(url).json()
                    if job["state"]["status"] in ("failed","completed"):
                        break
                    time.sleep(.2)
                assert job["state"]["status"]=="completed",job["state"]
                assert len(job["results"])==1
                assert fresh.get(url+"/download/inference.pt").is_success
            print("HTTP import, worker pause/resume, reconnect, export passed",sys.platform)
        finally:
            server.terminate()
            server.wait(timeout=30)


if __name__=="__main__":
    main()
