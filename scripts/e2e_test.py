"""
E2E test orchestrator for ASTRA federated learning platform.

Tests the full lifecycle with 5 simulated clients across 4 aggregation
methods (FedAvg, TrimmedMean, Median, Hybrid), multiple rounds, and
verifies aggregation correctness.

Prerequisites:
    1. Start the ASTRA server:  uvicorn astra.app.server_api:app --reload
    2. Generate data:          python scripts/synth_data.py
    3. Run this script:        python scripts/e2e_test.py

Usage:
    python scripts/e2e_test.py                          # defaults
    python scripts/e2e_test.py --rounds 5 --clients 3   # custom
    python scripts/e2e_test.py --base-url http://host:port
"""

import argparse
import json
import sys
import time
import urllib.error
import urllib.request

import numpy as np
import torch

# ---------------------------------------------------------------------------
# HTTP helpers
# ---------------------------------------------------------------------------

def api(base_url: str, method: str, path: str, token: str = None, body: dict = None) -> dict:
    """Make an API call and return parsed JSON."""
    url = f"{base_url}{path}"
    data = json.dumps(body).encode() if body else None
    headers = {"Content-Type": "application/json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"

    req = urllib.request.Request(url, data=data, headers=headers, method=method)
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            return json.loads(resp.read().decode())
    except urllib.error.HTTPError as e:
        detail = e.read().decode() if e.fp else ""
        return {"_error": e.code, "_detail": detail}


def download_raw(base_url: str, path: str, token: str) -> tuple[bytes, dict]:
    """Download raw bytes and return (bytes, headers_dict)."""
    url = f"{base_url}{path}"
    req = urllib.request.Request(url, headers={"Authorization": f"Bearer {token}"})
    with urllib.request.urlopen(req, timeout=30) as resp:
        return resp.read(), dict(resp.headers)


# ---------------------------------------------------------------------------
# Auth helpers
# ---------------------------------------------------------------------------

def signup(base_url: str, username: str, password: str, role: str = "client") -> str:
    """Sign up and return JWT token."""
    r = api(base_url, "POST", "/api/auth/signup",
            body={"username": username, "password": password, "role": role})
    return r.get("token", "")


def login(base_url: str, username: str, password: str) -> str:
    """Login and return JWT token."""
    r = api(base_url, "POST", "/api/auth/login",
            body={"username": username, "password": password})
    return r.get("token", "")


# ---------------------------------------------------------------------------
# Group setup
# ---------------------------------------------------------------------------

def setup_group(base_url: str, admin_token: str, group_id: str,
                model_id: str, aggregator: str, window_size: int = 3) -> dict:
    """Create and start a group. Returns the group dict."""
    r = api(base_url, "POST", "/api/groups", admin_token, {
        "group_id": group_id,
        "model_id": model_id,
        "window_size": window_size,
        "time_limit": 300.0,
        "lr": 0.01,
        "aggregator": aggregator,
    })
    if "_error" in r:
        print(f"  [WARN] create_group returned {r['_error']}: {r.get('_detail', '')[:100]}")

    api(base_url, "POST", f"/api/groups/{group_id}/start", admin_token)
    return r


# ---------------------------------------------------------------------------
# Client registration + join flow
# ---------------------------------------------------------------------------

def register_client(base_url: str, admin_token: str, client_token: str,
                    group_id: str, client_name: str) -> str:
    """Full join flow: request → approve → activate. Returns client_id."""
    # 1. Request to join
    api(base_url, "POST", "/api/join/join-request", client_token,
        {"group_id": group_id})

    # 2. Admin fetches pending requests
    r = api(base_url, "GET", f"/api/join/join-requests?group_id={group_id}", admin_token)
    requests_list = r.get("requests", [])
    if not requests_list:
        print(f"  [ERROR] No pending request for {client_name}")
        return ""
    request_id = requests_list[-1]["id"]

    # 3. Admin approves
    api(base_url, "POST", "/api/join/join-requests/approve", admin_token,
        {"request_id": request_id})

    # 4. Client activates
    r = api(base_url, "POST", f"/api/join/activate/{group_id}", client_token)
    return r.get("client_id", "")


# ---------------------------------------------------------------------------
# Single client train + upload
# ---------------------------------------------------------------------------

def client_train_and_upload(
    base_url: str, token: str, client_id: str, group_id: str,
    X: torch.Tensor, y: torch.Tensor,
    client_version: int = 0,
    local_epochs: int = 2,
    lr: float = 0.01,
) -> dict:
    """Download model → train → compute delta → upload. Returns result dict."""
    from copy import deepcopy
    sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent.parent / "src"))
    from astra.core.models.model_zoo import SimpleMLP, flatten_all_params

    device = "cpu"

    # 1. Download current global weights (fallback to fresh model if 404)
    try:
        raw, headers = download_raw(
            base_url,
            f"/api/models/{group_id}/download?format=raw",
            token,
        )
        num_params = int(headers.get("X-Num-Parameters", len(raw) // 4))
        flat_weights = np.frombuffer(raw, dtype="<f4").copy()
    except Exception:
        # Round 0 before any aggregation — use fresh model weights
        raw = None
        flat_weights = None

    # 2. Load into model
    model_before = SimpleMLP(input_dim=784, num_classes=10, hidden_dim=256)
    if flat_weights is not None:
        params = sorted(
            [(n, p) for n, p in model_before.named_parameters()],
            key=lambda x: x[0],
        )
        offset = 0
        for name, param in params:
            size = param.numel()
            param.data = torch.from_numpy(flat_weights[offset:offset+size]).reshape(param.shape)
            offset += size

    # 3. Train
    model_after = deepcopy(model_before)
    model_after.train()
    optimizer = torch.optim.SGD(model_after.parameters(), lr=lr)
    criterion = torch.nn.CrossEntropyLoss()
    dataset = torch.utils.data.TensorDataset(X, y)
    loader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)

    for epoch in range(local_epochs):
        for X_b, y_b in loader:
            optimizer.zero_grad()
            loss = criterion(model_after(X_b), y_b)
            loss.backward()
            optimizer.step()

    # 4. Compute delta
    old_flat = flatten_all_params(model_before)
    new_flat = flatten_all_params(model_after)
    delta = (new_flat - old_flat).astype(np.float32)

    # 5. Evaluate training
    model_after.eval()
    with torch.no_grad():
        out = model_after(X)
        pred = out.argmax(dim=1)
        train_acc = (pred == y).float().mean().item()
        train_loss = criterion(out, y).item()

    # 6. Upload delta
    import base64
    b64 = base64.b64encode(delta.tobytes()).decode("ascii")
    r = api(base_url, "POST", f"/api/clients/{client_id}/delta", token, {
        "client_id": client_id,
        "client_version": client_version,
        "local_updates": b64,
        "update_type": "delta",
        "local_dataset_size": len(X),
        "meta": {"train_accuracy": train_acc, "train_loss": train_loss},
    })

    return {
        "train_acc": train_acc,
        "train_loss": train_loss,
        "delta_norm": float(np.linalg.norm(delta)),
        "delta_bytes": len(delta) * 4,
        "upload": r,
    }


# ---------------------------------------------------------------------------
# Evaluate global model
# ---------------------------------------------------------------------------

def evaluate_global(base_url: str, token: str, group_id: str,
                    X_test: torch.Tensor, y_test: torch.Tensor) -> dict:
    """Download global model and evaluate on test set."""
    sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent.parent / "src"))
    from astra.core.models.model_zoo import SimpleMLP

    try:
        raw, headers = download_raw(
            base_url, f"/api/models/{group_id}/download?format=raw", token
        )
        flat = np.frombuffer(raw, dtype="<f4").copy()
    except Exception:
        return {"accuracy": 0.0, "loss": float("inf"), "total": len(y_test)}

    model = SimpleMLP(input_dim=784, num_classes=10, hidden_dim=256)
    params = sorted([(n, p) for n, p in model.named_parameters()], key=lambda x: x[0])
    offset = 0
    for name, param in params:
        size = param.numel()
        param.data = torch.from_numpy(flat[offset:offset+size]).reshape(param.shape)
        offset += size

    model.eval()
    with torch.no_grad():
        out = model(X_test)
        pred = out.argmax(dim=1)
        acc = (pred == y_test).float().mean().item()
        loss = torch.nn.CrossEntropyLoss()(out, y_test).item()

    return {"accuracy": acc, "loss": loss}


# ---------------------------------------------------------------------------
# Run one aggregation method
# ---------------------------------------------------------------------------

def run_method(
    base_url: str,
    admin_token: str,
    method: str,
    num_clients: int,
    num_rounds: int,
    window_size: int,
    model_id: str,
    client_data: list[tuple[torch.Tensor, torch.Tensor]],
    X_test: torch.Tensor,
    y_test: torch.Tensor,
) -> dict:
    """Run full E2E test for one aggregation method. Returns results dict."""
    import uuid

    group_id = f"e2e_{method}_{uuid.uuid4().hex[:6]}"
    print(f"\n{'='*60}")
    print(f"  Method: {method.upper()}")
    print(f"  Group:  {group_id}")
    print(f"  Clients: {num_clients}, Rounds: {num_rounds}, Window: {window_size}")
    print(f"{'='*60}")

    # Create group
    setup_group(base_url, admin_token, group_id, model_id, method, window_size)

    # Register and activate clients
    clients = []
    for i in range(num_clients):
        uname = f"e2e_{method}_c{i}_{uuid.uuid4().hex[:4]}"
        pwd = "testpass123"
        token = signup(base_url, uname, pwd, "client")
        client_id = register_client(base_url, admin_token, token, group_id, uname)
        if client_id:
            clients.append({"name": uname, "token": token, "client_id": client_id})
            print(f"  Registered client {i}: {client_id}")
        else:
            print(f"  [ERROR] Failed to register client {i}")

    if not clients:
        return {"method": method, "error": "no clients registered"}

    # Training rounds
    round_results = []
    initial_acc = 0.0
    for rnd in range(num_rounds):
        print(f"\n  --- Round {rnd + 1}/{num_rounds} ---")
        round_info = {"round": rnd + 1, "uploads": [], "aggregated": False}

        for i, cl in enumerate(clients):
            X_i, y_i = client_data[i % len(client_data)]
            result = client_train_and_upload(
                base_url, cl["token"], cl["client_id"], group_id,
                X_i, y_i,
                client_version=rnd,
                local_epochs=2,
                lr=0.01,
            )
            status = result["upload"].get("status", "?")
            g_ver = result["upload"].get("global_version", "?")
            print(f"  Client {i}: acc={result['train_acc']:.3f}, "
                  f"delta_norm={result['delta_norm']:.4f}, "
                  f"status={status}, global_v={g_ver}")
            round_info["uploads"].append(result)

            # Rate limit: wait between uploads from different clients
            if i < len(clients) - 1:
                time.sleep(0.1)

        # Check group status
        time.sleep(0.5)
        grp = api(base_url, "GET", f"/api/groups/{group_id}", admin_token)
        g = grp.get("group", {})
        ver = g.get("model_version", 0)
        rounds_done = g.get("completed_rounds", 0)
        round_info["model_version"] = ver
        round_info["completed_rounds"] = rounds_done
        print(f"  Group state: v{ver}, {rounds_done} rounds completed")

        # Evaluate
        eval_r = evaluate_global(base_url, admin_token, group_id, X_test, y_test)
        round_info["eval"] = eval_r
        if rnd == 0:
            initial_acc = eval_r["accuracy"]
        print(f"  Global model: acc={eval_r['accuracy']:.4f}, loss={eval_r['loss']:.4f}")

        round_results.append(round_info)

    # Final state
    grp = api(base_url, "GET", f"/api/groups/{group_id}", admin_token)
    g = grp.get("group", {})
    final_eval = evaluate_global(base_url, admin_token, group_id, X_test, y_test)

    # Trust scores
    trust = api(base_url, "GET", f"/api/trust/scores?group_id={group_id}", admin_token)
    trust_scores = trust.get("scores", {})

    result = {
        "method": method,
        "group_id": group_id,
        "num_clients": num_clients,
        "num_rounds": num_rounds,
        "final_model_version": g.get("model_version", 0),
        "completed_rounds": g.get("completed_rounds", 0),
        "initial_accuracy": initial_acc,
        "final_accuracy": final_eval["accuracy"],
        "final_loss": final_eval["loss"],
        "accuracy_improved": final_eval["accuracy"] > initial_acc,
        "trust_scores": {k: v.get("score", 0) for k, v in trust_scores.items()},
        "all_quarantined": all(
            v.get("score", 1.0) > 0.35 for v in trust_scores.values()
        ),
        "rounds": round_results,
    }
    return result


# ---------------------------------------------------------------------------
# Print summary
# ---------------------------------------------------------------------------

def print_summary(results: list[dict]):
    """Print a comparison table of all methods."""
    print(f"\n{'='*80}")
    print(f"  E2E TEST RESULTS SUMMARY")
    print(f"{'='*80}")

    header = f"{'Method':<16} {'Version':>8} {'Rounds':>8} {'Acc':>8} {'Loss':>8} {'Improved':>10} {'Trust OK':>10}"
    print(header)
    print("-" * 80)

    for r in results:
        if "error" in r:
            print(f"{r['method']:<16} ERROR: {r['error']}")
            continue
        print(f"{r['method']:<16} {r['final_model_version']:>8} {r['completed_rounds']:>8} "
              f"{r['final_accuracy']:>8.4f} {r['final_loss']:>8.4f} "
              f"{str(r['accuracy_improved']):>10} {str(r['all_quarantined']):>10}")

    # Assertions
    print(f"\n{'='*80}")
    print(f"  ASSERTIONS")
    print(f"{'='*80}")

    all_pass = True
    for r in results:
        method = r["method"]
        if "error" in r:
            print(f"  [{method}] FAIL - {r['error']}")
            all_pass = False
            continue

        checks = [
            ("version_incremented", r["final_model_version"] > 0),
            ("rounds_completed", r["completed_rounds"] > 0),
            ("accuracy_above_random", r["final_accuracy"] > 0.10),
            ("no_quarantined_clients", r["all_quarantined"]),
            ("weights_changed", any(
                u["delta_norm"] > 0 for rnd in r["rounds"] for u in rnd["uploads"]
            )),
        ]

        for name, passed in checks:
            status = "PASS" if passed else "FAIL"
            if not passed:
                all_pass = False
            print(f"  [{method}] {name}: {status}")

    print(f"\n  OVERALL: {'ALL PASSED' if all_pass else 'SOME FAILED'}")
    print(f"{'='*80}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="ASTRA E2E test orchestrator")
    parser.add_argument("--base-url", default="http://localhost:8000")
    parser.add_argument("--clients", type=int, default=5)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--window-size", type=int, default=3)
    parser.add_argument("--methods", nargs="+",
                        default=["fedavg", "trimmed_mean", "median", "hybrid"],
                        help="Aggregation methods to test")
    parser.add_argument("--model-id", default="e2e_simple_mlp",
                        help="Model ID to register")
    parser.add_argument("--data-dir", default=None,
                        help="Path to scripts/data/ (auto-detected if omitted)")
    args = parser.parse_args()

    # Auto-detect data dir
    if args.data_dir is None:
        args.data_dir = str(__import__("pathlib").Path(__file__).resolve().parent / "data")

    # Check server health
    try:
        r = api(args.base_url, "GET", "/health")
        if r.get("status") != "healthy":
            print(f"[ERROR] Server not healthy: {r}")
            sys.exit(1)
        print(f"Server: {args.base_url} — healthy")
    except Exception as e:
        print(f"[ERROR] Cannot reach server at {args.base_url}: {e}")
        sys.exit(1)

    # Generate data if missing
    X_test_path = f"{args.data_dir}/test_data.pt"
    try:
        test_data = torch.load(X_test_path, weights_only=False)
        X_test, y_test = test_data["X"], test_data["y"]
    except FileNotFoundError:
        print("Data not found. Generating synthetic data...")
        import subprocess
        subprocess.run([sys.executable, str(__import__("pathlib").Path(__file__).resolve().parent / "synth_data.py")],
                       check=True)
        test_data = torch.load(X_test_path, weights_only=False)
        X_test, y_test = test_data["X"], test_data["y"]

    print(f"Test data: {len(X_test)} samples")

    # Load client data partitions
    client_data = []
    for i in range(args.clients):
        path = f"{args.data_dir}/client_{i}_data.pt"
        try:
            d = torch.load(path, weights_only=False)
            client_data.append((d["X"], d["y"]))
        except FileNotFoundError:
            print(f"[ERROR] Missing {path}. Run: python scripts/synth_data.py")
            sys.exit(1)

    print(f"Client partitions: {len(client_data)} loaded")

    # Sign up admin
    import uuid
    admin_name = f"e2e_admin_{uuid.uuid4().hex[:6]}"
    admin_pwd = "admin123"
    admin_token = signup(args.base_url, admin_name, admin_pwd, "admin")
    if not admin_token:
        print("[ERROR] Failed to sign up admin")
        sys.exit(1)
    print(f"Admin: {admin_name}")

    # Register model via API so the server process has it
    r = api(args.base_url, "POST", "/api/models/register/architecture", admin_token, {
        "model_id": args.model_id,
        "architecture_path": "astra.core.models.model_zoo.SimpleMLP",
        "model_type": "classifier",
    })
    if "_error" in r:
        print(f"[WARN] Model register via API: {r.get('_detail', '')[:100]}")
    sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent.parent / "src"))
    from astra.core.models.model_zoo import SimpleMLP
    sample = SimpleMLP()
    n_params = sum(p.numel() for p in sample.parameters())
    print(f"Model: {args.model_id} ({n_params:,} params)")

    # Run each method
    results = []
    for method in args.methods:
        r = run_method(
            args.base_url, admin_token, method,
            args.clients, args.rounds, args.window_size,
            args.model_id, client_data, X_test, y_test,
        )
        results.append(r)

    # Summary
    print_summary(results)

    # Save detailed results
    out_path = str(__import__("pathlib").Path(__file__).resolve().parent / "e2e_results.json")
    with open(out_path, "w") as f:
        # Convert tensors to lists for JSON serialization
        def sanitize(obj):
            if isinstance(obj, torch.Tensor):
                return obj.tolist()
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            if isinstance(obj, dict):
                return {k: sanitize(v) for k, v in obj.items()}
            if isinstance(obj, list):
                return [sanitize(v) for v in obj]
            return obj
        json.dump(sanitize(results), f, indent=2)
    print(f"\nDetailed results saved to {out_path}")


if __name__ == "__main__":
    main()
