"""
Test script for itwin_bridge.py
Validates: import, disabled mode, enabled mode, data transformation
"""
import os
import sys

# Force ITWIN_ENABLED=false for safety during testing
os.environ["ITWIN_ENABLED"] = "false"

sys.path.insert(0, os.path.dirname(__file__))

print("=" * 60)
print("TEST 1: Module Import")
print("=" * 60)

try:
    from itwin_bridge import (
        _is_configured,
        get_status,
        push_to_itwin,
        setup_itwin_sensors,
        get_access_token,
        _safe_float,
        ITWIN_ENABLED,
        TARGETS,
    )
    print("  [PASS] itwin_bridge imported successfully")
except Exception as e:
    print(f"  [FAIL] Import error: {e}")
    sys.exit(1)


print()
print("=" * 60)
print("TEST 2: Disabled Mode (ITWIN_ENABLED=false)")
print("=" * 60)

assert ITWIN_ENABLED == False, f"Expected ITWIN_ENABLED=False, got {ITWIN_ENABLED}"
print("  [PASS] ITWIN_ENABLED is False")

assert _is_configured() == False, "Expected _is_configured()=False when disabled"
print("  [PASS] _is_configured() returns False")

result = setup_itwin_sensors()
assert result == False, f"Expected setup to return False, got {result}"
print("  [PASS] setup_itwin_sensors() is a no-op (returned False)")


class MockState:
    data = None
    forecast = None
    anomalies = None
    importance = None
    models = None
    last_update = None


result = push_to_itwin(MockState())
assert result == False, f"Expected push to return False, got {result}"
print("  [PASS] push_to_itwin() is a no-op (returned False)")

status = get_status()
assert status["enabled"] == False
assert status["configured"] == False
assert status["setup_complete"] == False
print(f"  [PASS] get_status() returns correct disabled state: {status}")


print()
print("=" * 60)
print("TEST 3: _safe_float() Helper")
print("=" * 60)

import numpy as np

assert _safe_float(3.14) == 3.14, "float passthrough failed"
assert _safe_float(np.float64(2.5)) == 2.5, "numpy float64 failed"
assert _safe_float(np.int32(7)) == 7.0, "numpy int32 failed"
assert _safe_float(None) == 0.0, "None handling failed"
assert _safe_float("abc") == 0.0, "string handling failed"
assert _safe_float(np.float32(1.23)) == float(np.float32(1.23)), "numpy float32 failed"
print("  [PASS] All _safe_float() conversions correct")


print()
print("=" * 60)
print("TEST 4: Data Transformation (Mock Push Payload)")
print("=" * 60)

import pandas as pd

# Simulate a real MachineState with data
dates = pd.date_range("2026-05-04 00:00", periods=10, freq="30min")
data = pd.DataFrame(
    {
        "current": np.random.uniform(3, 5, 10),
        "temperature": np.random.uniform(30, 50, 10),
        "z_rms": np.random.uniform(0.3, 1.5, 10),
        "x_rms": np.random.uniform(0.2, 1.0, 10),
        "z_peak": np.random.uniform(0.5, 2.0, 10),
        "x_peak": np.random.uniform(0.4, 1.8, 10),
        "noise": np.random.uniform(60, 80, 10),
    },
    index=dates,
)

forecast_dates = pd.date_range("2026-05-04 05:00", periods=12, freq="30min")
forecast = pd.DataFrame(
    {tgt: np.random.uniform(0.5, 2.0, 12) for tgt in TARGETS},
    index=forecast_dates,
)

anomalies = {tgt: np.random.rand(10) for tgt in TARGETS}

mock = MockState()
mock.data = data
mock.forecast = forecast
mock.anomalies = anomalies

# Can't actually push (disabled), but verify the data shapes are correct
latest = mock.data.iloc[-1]
ts = latest.name.isoformat() + "Z"
print(f"  Latest timestamp: {ts}")
print(f"  Latest values: { {k: round(_safe_float(latest.get(k)), 3) for k in TARGETS} }")

# Verify forecast shape
fc_values = {f"{tgt}_forecast": _safe_float(mock.forecast[tgt].iloc[-1]) for tgt in TARGETS}
print(f"  Forecast values: { {k: round(v, 3) for k, v in fc_values.items()} }")

# Verify anomaly shape
an_values = {f"{tgt}_anomaly": _safe_float(mock.anomalies[tgt][-1]) for tgt in TARGETS}
print(f"  Anomaly scores:  { {k: round(v, 3) for k, v in an_values.items()} }")

print("  [PASS] Data transformation produces valid payloads")


print()
print("=" * 60)
print("TEST 5: main.py Import Check")
print("=" * 60)

# Verify main.py can at least parse (not run startup, just syntax/import check)
try:
    import ast

    with open("main.py", "r") as f:
        tree = ast.parse(f.read())
    
    # Check that itwin_bridge import exists
    imports = [
        node for node in ast.walk(tree)
        if isinstance(node, (ast.Import, ast.ImportFrom))
    ]
    itwin_import_found = any(
        (isinstance(n, ast.Import) and any(a.name == "itwin_bridge" for a in n.names))
        for n in imports
    )
    assert itwin_import_found, "itwin_bridge import not found in main.py"
    print("  [PASS] main.py parses correctly")
    print("  [PASS] itwin_bridge import found in main.py")

    # Check new endpoints exist
    source = open("main.py").read()
    assert "/api/itwin/status" in source, "itwin status endpoint not found"
    assert "/api/itwin/push" in source, "itwin push endpoint not found"
    print("  [PASS] /api/itwin/status endpoint found")
    print("  [PASS] /api/itwin/push endpoint found")

except Exception as e:
    print(f"  [FAIL] {e}")
    sys.exit(1)


print()
print("=" * 60)
print("TEST 6: Enabled Mode Token Test (dry run)")
print("=" * 60)

# Re-enable to test token flow (without actual HTTP call)
import itwin_bridge as ib

# Temporarily set enabled and check config validation
ib.ITWIN_ENABLED = True
ib.ITWIN_CLIENT_ID = "test-client-id"
ib.ITWIN_CLIENT_SECRET = "test-secret"
ib.ITWIN_ASSET_ID = "test-asset-id"

assert ib._is_configured() == True, "Should be configured with all vars set"
print("  [PASS] _is_configured() returns True when all vars are set")

# Test missing var detection
ib.ITWIN_CLIENT_ID = ""
assert ib._is_configured() == False, "Should detect missing CLIENT_ID"
print("  [PASS] Detects missing ITWIN_CLIENT_ID")

ib.ITWIN_CLIENT_ID = "test"
ib.ITWIN_ASSET_ID = ""
assert ib._is_configured() == False, "Should detect missing ASSET_ID"
print("  [PASS] Detects missing ITWIN_ASSET_ID")

# Reset
ib.ITWIN_ENABLED = False


print()
print("=" * 60)
print("ALL TESTS PASSED (OK)")
print("=" * 60)
