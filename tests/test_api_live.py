"""
Live API tests: single predictions, SHAP observation, and batch timing.
Run with: python tests/test_api_live.py
Requires the API server to be running on localhost:8000.
"""
import time
import json
import requests

BASE = "http://localhost:8000"

# ---------------------------------------------------------------------------
# Representative test transactions
# ---------------------------------------------------------------------------
TRANSACTIONS = [
    {
        "label": "Low-risk small purchase",
        "TransactionID": 1001,
        "TransactionAmt": 29.95,
        "ProductCD": "W",
        "card1": 4921, "card2": 325.0, "card3": 150.0,
        "card4": "visa", "card5": 226.0, "card6": "debit",
        "addr1": 315.0, "addr2": 87.0,
        "P_emaildomain": "gmail.com", "R_emaildomain": "gmail.com",
        "C1": 1.0, "C2": 1.0, "C5": 0.0, "C8": 1.0, "C9": 1.0, "C12": 0.0,
        "D1": 0.0, "D3": 0.0, "D4": 0.0, "D10": 1.0, "D11": 0.0,
        "M1": "T", "M4": "M0", "M5": "T", "M6": "T", "M7": "F",
    },
    {
        "label": "High-risk large amount",
        "TransactionID": 1002,
        "TransactionAmt": 1200.0,
        "ProductCD": "H",
        "card1": 9999, "card2": 111.0, "card3": 150.0,
        "card4": "mastercard", "card5": 102.0, "card6": "credit",
        "addr1": 0.0, "addr2": 60.0,
        "P_emaildomain": "protonmail.com", "R_emaildomain": "anonymous.com",
        "C1": 1.0, "C2": 3.0, "C5": 3.0, "C8": 4.0, "C9": 0.0, "C12": 3.0,
        "D1": 1.0, "D3": 1.0, "D4": 1.0, "D10": 0.0, "D11": 0.0,
        "M1": "F", "M4": "M2", "M5": "F", "M6": "F", "M7": "F",
    },
    {
        "label": "Mid-range mixed signals",
        "TransactionID": 1003,
        "TransactionAmt": 149.99,
        "ProductCD": "C",
        "card1": 7123, "card2": 250.0, "card3": 150.0,
        "card4": "visa", "card5": 226.0, "card6": "credit",
        "addr1": 204.0, "addr2": 87.0,
        "P_emaildomain": "yahoo.com", "R_emaildomain": "gmail.com",
        "C1": 2.0, "C2": 2.0, "C5": 1.0, "C8": 2.0, "C9": 1.0, "C12": 0.0,
        "D1": 2.0, "D3": 0.0, "D4": 2.0, "D10": 3.0, "D11": 1.0,
        "M1": "T", "M4": "M1", "M5": "T", "M6": "F", "M7": "F",
    },
    {
        "label": "Minimal fields only",
        "TransactionID": 1004,
        "TransactionAmt": 75.0,
        "ProductCD": "R",
        "card1": 3500,
    },
    {
        "label": "Tiny amount edge case",
        "TransactionID": 1005,
        "TransactionAmt": 0.01,
        "ProductCD": "S",
        "card1": 1234, "card2": 100.0,
    },
]


def _predict(txn_payload: dict) -> tuple[dict, float]:
    """POST to /predict and return (response_json, latency_ms)."""
    t0 = time.perf_counter()
    r = requests.post(f"{BASE}/predict", json=txn_payload, timeout=30)
    ms = (time.perf_counter() - t0) * 1000
    r.raise_for_status()
    return r.json(), ms


# ---------------------------------------------------------------------------
# Single-transaction tests
# ---------------------------------------------------------------------------
def run_single_tests():
    print("=" * 70)
    print("SINGLE TRANSACTION SCORING TESTS")
    print("=" * 70)

    for txn in TRANSACTIONS:
        payload = {k: v for k, v in txn.items() if k != "label"}
        label = txn["label"]

        try:
            result, ms = _predict(payload)
        except Exception as exc:
            print(f"\n[{label}] FAILED: {exc}")
            continue

        shap_drivers = result.get("top_shap_features", [])
        print(f"\n[{label}]")
        print(f"  amt              : {payload['TransactionAmt']}")
        print(f"  fraud_probability: {result['fraud_probability']:.6f}")
        print(f"  decision         : {result['decision']}")
        print(f"  threshold_used   : {result['threshold_used']}")
        print(f"  redis_enriched   : {result['redis_enriched']}")
        if shap_drivers:
            print("  top SHAP drivers :")
            for d in shap_drivers:
                sign = "+" if d["shap_contribution"] > 0 else ""
                print(f"    {sign}{d['shap_contribution']:+.6f}  {d['feature']}")
        else:
            print("  top SHAP drivers : (unavailable)")
        print(f"  latency          : {ms:.1f} ms")


# ---------------------------------------------------------------------------
# Batch timing tests
# ---------------------------------------------------------------------------
def _build_batch(n: int) -> list[dict]:
    """Round-robin the 5 test transactions to fill a batch of size n."""
    base = [{k: v for k, v in t.items() if k != "label"} for t in TRANSACTIONS]
    batch = []
    for i in range(n):
        txn = dict(base[i % len(base)])
        txn["TransactionID"] = 2000 + i
        batch.append(txn)
    return batch


def run_batch_tests():
    print("\n" + "=" * 70)
    print("BATCH TIMING TESTS  (sequential /predict calls — no dedicated endpoint)")
    print("=" * 70)

    for batch_size in (10, 50, 100):
        batch = _build_batch(batch_size)
        t_start = time.perf_counter()
        errors = 0
        for txn in batch:
            try:
                _predict(txn)
            except Exception:
                errors += 1
        elapsed_ms = (time.perf_counter() - t_start) * 1000
        per_txn = elapsed_ms / batch_size

        print(
            f"\n  batch={batch_size:>3}  total={elapsed_ms:>7.1f} ms  "
            f"per-txn={per_txn:>6.1f} ms  errors={errors}"
        )

    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    # Confirm server is up
    try:
        health = requests.get(f"{BASE}/health", timeout=5).json()
        print(f"Health check: {json.dumps(health, indent=2)}\n")
    except Exception as exc:
        print(f"Server not reachable: {exc}")
        raise SystemExit(1)

    run_single_tests()
    run_batch_tests()
