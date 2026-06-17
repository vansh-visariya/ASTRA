"""End-to-end smoke test using a live uvicorn server."""

import base64
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import requests


def main():
    repo = Path(__file__).resolve().parent.parent
    env = os.environ.copy()
    env["PYTHONPATH"] = str(repo / "src")

    print("Starting uvicorn on :8767 ...")
    proc = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "astra.app.server_api:app", "--port", "8767"],
        cwd=repo,
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    try:
        base = "http://127.0.0.1:8767"
        for _ in range(30):
            try:
                if requests.get(f"{base}/health", timeout=1).status_code == 200:
                    break
            except Exception:
                time.sleep(0.5)

        admin_user = f"smoke_a_{os.urandom(4).hex()}"
        requests.post(
            f"{base}/api/auth/signup",
            json={"username": admin_user, "password": "testpass123", "role": "admin"},
        )
        admin = requests.post(
            f"{base}/api/auth/login", json={"username": admin_user, "password": "testpass123"}
        ).json()
        admin_tok = admin["token"]
        h = {"Authorization": f"Bearer {admin_tok}"}

        r = requests.post(
            f"{base}/api/models/register/architecture",
            json={
                "model_id": "smoke_mlp_e2e",
                "architecture_path": "astra.core.models.model_zoo.SimpleMLP",
                "model_type": "vision",
                "config": {},
            },
            headers=h,
        )
        print(f"register: {r.status_code}")
        assert r.status_code == 200

        gid = f"smoke_{os.urandom(4).hex()}"
        r = requests.post(
            f"{base}/api/groups",
            json={
                "group_id": gid,
                "model_id": "smoke_mlp_e2e",
                "window_size": 1,
                "time_limit": 60,
                "lr": 0.01,
                "aggregator": "fedavg",
            },
            headers=h,
        )
        assert r.status_code == 200, r.text
        requests.post(f"{base}/api/groups/{gid}/start", headers=h)

        client_user = f"smoke_c_{os.urandom(4).hex()}"
        requests.post(
            f"{base}/api/auth/signup",
            json={"username": client_user, "password": "testpass123", "role": "client"},
        )
        cl = requests.post(
            f"{base}/api/auth/login",
            json={"username": client_user, "password": "testpass123"},
        ).json()
        ch = {"Authorization": f"Bearer {cl['token']}"}

        requests.post(
            f"{base}/api/join/join-request", json={"group_id": gid}, headers=ch
        )
        pending = requests.get(
            f"{base}/api/join/join-requests?group_id={gid}", headers=h
        ).json()
        requests.post(
            f"{base}/api/join/join-requests/approve",
            json={"request_id": pending["requests"][0]["id"]},
            headers=h,
        )
        act = requests.post(f"{base}/api/join/activate/{gid}", headers=ch).json()
        cid = act["client_id"]

        delta = np.random.default_rng(1).standard_normal(73).astype("<f4").tobytes()
        b64 = base64.b64encode(delta).decode("ascii")
        r = requests.post(
            f"{base}/api/clients/{cid}/delta",
            json={
                "client_id": cid,
                "client_version": 0,
                "local_updates": b64,
                "update_type": "delta",
                "local_dataset_size": 100,
                "meta": {},
            },
            headers=ch,
        )
        print(f"upload: {r.status_code} {r.json()}")
        assert r.status_code == 200
        assert r.json()["status"] == "accepted"
        assert r.json()["global_version"] == 1

        gs = requests.get(f"{base}/api/groups/{gid}", headers=h).json()
        print(f"group model_version: {gs['group']['model_version']}")
        assert gs["group"]["model_version"] == 1
        assert gs["group"]["completed_rounds"] == 1

        r = requests.get(f"{base}/api/models/{gid}/download", headers=h)
        print(f"download: {r.status_code}, {len(r.content)} bytes")
        assert r.status_code == 200
        assert len(r.content) > 0

        print("\nSMOKE PASSED")
        return 0
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=3)
        except subprocess.TimeoutExpired:
            proc.kill()


if __name__ == "__main__":
    sys.exit(main())
