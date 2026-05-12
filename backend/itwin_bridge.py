"""
itwin_bridge.py — Bentley iTwin IoT Sensor Data Integration Bridge

Pushes raw sensor readings AND ML prediction outputs (forecasts, anomaly scores)
to the Bentley iTwin IoT platform via their REST Sensor Data API.

Runs inside the existing FastAPI process alongside APScheduler.
Gracefully degrades to a no-op when credentials are not configured.
"""

import os
import json
import time
import logging
import numpy as np
import requests
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger("itwin_bridge")
logger.setLevel(logging.INFO)
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("[%(levelname)s] iTwin Bridge: %(message)s"))
    logger.addHandler(handler)

# ===================== CONFIGURATION =====================

ITWIN_ENABLED = os.getenv("ITWIN_ENABLED", "false").lower() == "true"
ITWIN_CLIENT_ID = os.getenv("ITWIN_CLIENT_ID", "")
ITWIN_CLIENT_SECRET = os.getenv("ITWIN_CLIENT_SECRET", "")
ITWIN_ASSET_ID = os.getenv("ITWIN_ASSET_ID", "")

TOKEN_URL = "https://ims.bentley.com/connect/token"
INTEGRATE_URL = "https://api.bentley.com/sensor-data/integrations/integrate"
UPLOAD_URL = "https://api.bentley.com/sensor-data/data/upload"

SENSOR_IDS_FILE = os.path.join(os.path.dirname(__file__), "sensor_ids.json")

# Sensor targets (must match ml_engine.py TARGETS)
TARGETS = ["current", "temperature", "z_rms", "x_rms", "z_peak", "x_peak", "noise"]

# Units mapping for each target — full names required by Bentley API
UNITS = {
    "current": "ampere",
    "temperature": "celsius",
    "z_rms": "millimeter",
    "x_rms": "millimeter",
    "z_peak": "millimeter",
    "x_peak": "millimeter",
    "noise": "decibel",
}

# Metric key names as registered in Bentley UNKNOWN_METRICS (must match upload values keys)
METRIC_NAMES = {
    "current": "current",
    "temperature": "temperature",
    "z_rms": "conv_z_rms",
    "x_rms": "conv_x_rms",
    "z_peak": "z_peak",
    "x_peak": "x_peak",
    "noise": "noise",
}

# ===================== GLOBAL STATE =====================

_token_cache = {
    "access_token": None,
    "expires_at": 0,  # Unix timestamp
}

# Sensor IDs — loaded from sensor_ids.json at module init
_sensor_ids: dict = {}

_last_push_status = {
    "timestamp": None,
    "success": None,
    "message": "",
    "observations_sent": 0,
}

_setup_complete = False

# Tracks the data timestamp of the last successfully pushed row (deduplication)
_last_pushed_data_ts: str | None = None


# ===================== HELPERS =====================

def _is_configured() -> bool:
    """Check if all required Bentley credentials are present."""
    if not ITWIN_ENABLED:
        return False
    missing = []
    if not ITWIN_CLIENT_ID:
        missing.append("ITWIN_CLIENT_ID")
    if not ITWIN_CLIENT_SECRET:
        missing.append("ITWIN_CLIENT_SECRET")
    if not ITWIN_ASSET_ID:
        missing.append("ITWIN_ASSET_ID")
    if missing:
        logger.warning(f"iTwin integration disabled — missing env vars: {', '.join(missing)}")
        return False
    return True


def _safe_float(val) -> float:
    """Safely convert a value to float, handling numpy types."""
    if val is None:
        return 0.0
    if isinstance(val, (np.integer, np.floating)):
        return float(val)
    try:
        return float(val)
    except (TypeError, ValueError):
        return 0.0


# ===================== SENSOR ID FILE PERSISTENCE =====================

def _load_sensor_ids_from_file() -> dict:
    try:
        with open(SENSOR_IDS_FILE, "r") as f:
            data = json.load(f)
        if data.get("registered") and isinstance(data.get("sensor_ids"), dict):
            return data["sensor_ids"]
    except (FileNotFoundError, json.JSONDecodeError, OSError) as e:
        logger.debug(f"Could not load sensor_ids.json: {e}")
    return {}


def _save_sensor_ids_to_file(ids: dict) -> None:
    payload = {"registered": True, "sensor_ids": ids}
    try:
        with open(SENSOR_IDS_FILE, "w") as f:
            json.dump(payload, f, indent=2)
            f.flush()
        logger.info(f"Sensor IDs saved to {SENSOR_IDS_FILE}")
    except OSError as e:
        logger.error(f"Failed to save sensor IDs: {e}")


def _parse_integrate_response(data: dict) -> dict:
    """Extract NAME→sensorId mapping from integrate API response."""
    ids: dict = {}
    try:
        for dev_block in data["integration"].get("devices", []):
            for sensor in dev_block.get("sensors", []):
                name = sensor.get("props", {}).get("NAME", "")
                sensor_id = sensor.get("id", "")
                if name and sensor_id:
                    ids[name] = sensor_id
    except (KeyError, TypeError) as e:
        logger.error(f"Failed to parse integrate response: {e}")
    return ids


# Load persisted sensor IDs at import time — fast path, no API call needed.
_sensor_ids = _load_sensor_ids_from_file()
if _sensor_ids:
    _setup_complete = True
    logger.info(f"Loaded {len(_sensor_ids)} sensor IDs from file — setup complete.")


# ===================== OAUTH TOKEN MANAGEMENT =====================

def get_access_token() -> str | None:
    """
    Obtain a Bentley OAuth2 access token using client_credentials grant.
    Caches the token and refreshes 5 minutes before expiry.
    """
    now = time.time()

    # Return cached token if still valid (with 300s buffer)
    if _token_cache["access_token"] and now < (_token_cache["expires_at"] - 300):
        return _token_cache["access_token"]

    logger.info("Requesting new Bentley access token...")

    try:
        resp = requests.post(
            TOKEN_URL,
            data={
                "grant_type": "client_credentials",
                "client_id": ITWIN_CLIENT_ID,
                "client_secret": ITWIN_CLIENT_SECRET,
                "scope": "itwin-platform",
            },
            headers={"Content-Type": "application/x-www-form-urlencoded"},
            timeout=30,
        )

        if resp.status_code != 200:
            logger.error(f"Token request failed ({resp.status_code}): {resp.text[:500]}")
            return None

        data = resp.json()
        _token_cache["access_token"] = data["access_token"]
        _token_cache["expires_at"] = now + data.get("expires_in", 3600)

        logger.info(f"Access token acquired (expires in {data.get('expires_in', '?')}s)")
        return _token_cache["access_token"]

    except Exception as e:
        logger.error(f"Token request exception: {e}")
        return None


def _auth_headers() -> dict | None:
    """Build authorization headers. Returns None if token unavailable."""
    token = get_access_token()
    if not token:
        return None
    return {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
    }


# ===================== SENSOR REGISTRATION =====================

def setup_itwin_sensors() -> bool:
    """
    Ensure sensors are registered in Bentley iTwin IoT.

    Fast path (normal): sensor_ids.json has registered=true → returns True immediately.
    Slow path (first deploy): calls the integrate API once, parses + saves the response.
    """
    global _setup_complete, _sensor_ids

    if not _is_configured():
        logger.info("iTwin integration is not configured — skipping sensor setup.")
        return False

    # Fast path: module-level init already loaded IDs from file
    if _setup_complete and _sensor_ids:
        return True

    # Second attempt: file may have been written between import and this call
    ids = _load_sensor_ids_from_file()
    if ids:
        _sensor_ids = ids
        _setup_complete = True
        return True

    # API path: first-time registration
    logger.info("No existing sensor IDs found — calling integrate API (one-time)...")
    headers = _auth_headers()
    if not headers:
        logger.error("Cannot call integrate API — no access token.")
        return False

    payload = {
        "integration": {
            "changeState": "new",
            "devices": [{
                "changeState": "new",
                "props": {"INTEGRATION_ID": "IMPORT_DEVICE_SDE", "NAME": "PM-Conveyor-Device"},
                "sensors": [
                    {
                        "changeState": "new",
                        "props": {
                            "INTEGRATION_ID": "GENERIC_SENSOR_SDE",
                            "NAME": name,
                            "UNKNOWN_UNITS": {"0": UNITS[name]},
                            "UNKNOWN_METRICS": {"0": METRIC_NAMES[name]},
                        },
                    }
                    for name in TARGETS
                ],
            }],
        }
    }

    try:
        resp = requests.post(
            INTEGRATE_URL,
            json=payload,
            headers=headers,
            params={"iTwinId": ITWIN_ASSET_ID},
            timeout=60,
        )
    except Exception as e:
        logger.error(f"Integrate API request failed: {e}")
        return False

    if resp.status_code not in (200, 201):
        logger.error(f"Integrate API returned {resp.status_code}: {resp.text[:500]}")
        return False

    ids = _parse_integrate_response(resp.json())
    if not ids:
        logger.error("Integrate API response contained no parseable sensor IDs.")
        return False

    _sensor_ids = ids
    _setup_complete = True
    _save_sensor_ids_to_file(ids)
    logger.info(f"Sensor registration complete — {len(ids)} sensors saved.")
    return True


# ===================== DATA PUSH =====================

def push_to_itwin(state) -> bool:
    """
    Push the latest raw sensor readings to Bentley iTwin IoT via data/upload.

    Skips the push if the data timestamp matches the last successful upload
    (deduplication — scheduler runs every 5 min, sensor data updates every 30 min).

    Returns True if push succeeded or was skipped (duplicate), False on error.
    """
    global _last_push_status, _last_pushed_data_ts

    if not _is_configured():
        return False

    if not _setup_complete:
        logger.info("Sensor setup not complete — attempting setup before push...")
        if not setup_itwin_sensors():
            logger.warning("Sensor setup still not complete — skipping push.")
            return False

    if state.data is None:
        logger.warning("No data available — skipping push.")
        return False

    # Deduplication: skip if this data timestamp was already uploaded.
    # Use the same ISO format as ts_str so the comparison is reliable.
    try:
        _last_ts = state.data.index[-1]
        if hasattr(_last_ts, "isoformat"):
            _candidate_ts = _last_ts.isoformat()
            if "+" not in _candidate_ts and "Z" not in _candidate_ts:
                _candidate_ts += "Z"
        else:
            _candidate_ts = str(_last_ts)
        if _candidate_ts == _last_pushed_data_ts:
            logger.debug(f"Skipping push — data timestamp {_candidate_ts} already uploaded.")
            return True
    except Exception:
        pass

    headers = _auth_headers()
    if not headers:
        logger.error("Push failed — could not obtain access token.")
        _last_push_status = {
            "timestamp": datetime.now().isoformat(),
            "success": False,
            "message": "Authentication failed",
            "observations_sent": 0,
        }
        return False

    try:
        latest = state.data.iloc[-1]
        features = latest.to_dict()
        # Use the sensor data timestamp in UTC ISO format
        timestamp = latest.name
        if hasattr(timestamp, "isoformat"):
            ts_str = timestamp.isoformat()
            # Ensure UTC format for Bentley
            if "+" not in ts_str and "Z" not in ts_str:
                ts_str += "Z"
        else:
            ts_str = str(timestamp)

        observations = []

        # Build one observation per sensor using the registered metric key name
        for tgt in TARGETS:
            sid = _sensor_ids.get(tgt)
            if not sid:
                continue
            metric_key = METRIC_NAMES[tgt]
            observations.append({
                "sensorId": sid,
                "timestamp": ts_str,
                "values": {metric_key: _safe_float(features.get(tgt, 0))},
            })

        # --- Upload to Bentley ---
        if not observations:
            logger.warning("No observations built — nothing to push.")
            _last_push_status = {
                "timestamp": datetime.now().isoformat(),
                "success": False,
                "message": "No observations built (sensor IDs not discovered?)",
                "observations_sent": 0,
            }
            return False

        resp = requests.post(
            UPLOAD_URL,
            json={"observations": observations},
            headers=headers,
            timeout=30,
        )

        if resp.status_code in (200, 201, 202, 204):
            logger.info(
                f"Push successful — {len(observations)} observations uploaded "
                f"(status {resp.status_code})"
            )
            _last_push_status = {
                "timestamp": datetime.now().isoformat(),
                "success": True,
                "message": f"Uploaded {len(observations)} observations",
                "observations_sent": len(observations),
            }
            _last_pushed_data_ts = ts_str
            return True
        else:
            logger.error(f"Push failed ({resp.status_code}): {resp.text[:500]}")
            _last_push_status = {
                "timestamp": datetime.now().isoformat(),
                "success": False,
                "message": f"HTTP {resp.status_code}: {resp.text[:200]}",
                "observations_sent": 0,
            }
            return False

    except Exception as e:
        logger.error(f"Push exception: {e}")
        _last_push_status = {
            "timestamp": datetime.now().isoformat(),
            "success": False,
            "message": str(e),
            "observations_sent": 0,
        }
        return False


# ===================== STATUS QUERY =====================

def debug_bentley_api() -> dict:
    """Debug endpoint to fetch existing nodes, devices, and sensors."""
    if not _is_configured():
        return {"error": "Not configured"}
    headers = _auth_headers()
    if not headers:
        return {"error": "No token"}
    
    result = {}
    try:
        # Get Nodes
        r1 = requests.get(f"https://api.bentley.com/sensor-data/integrations/nodes?iTwinId={ITWIN_ASSET_ID}", headers=headers, timeout=30)
        
        if r1.status_code != 200:
            return {"error": f"Failed to get nodes: {r1.text}"}
            
        nodes_data = r1.json()
        result["nodes_list"] = nodes_data
        result["detailed_nodes"] = {}
        
        nodes = nodes_data.get("nodes", [])
        for n in nodes:
            node_id = n.get("id")
            if not node_id:
                continue
                
            # For each node, get its full integration tree
            r2 = requests.post(
                INTEGRATE_URL,
                json={"integration": {"nodeId": node_id}},
                headers=headers,
                params={"iTwinId": ITWIN_ASSET_ID},
                timeout=30
            )
            
            if r2.status_code == 200:
                result["detailed_nodes"][node_id] = r2.json()
            else:
                result["detailed_nodes"][node_id] = {"error": r2.text}
                
    except Exception as e:
        result["error"] = str(e)
        
    return result

def get_status() -> dict:
    """Return the current bridge status for the /api/itwin/status endpoint."""
    return {
        "enabled": ITWIN_ENABLED,
        "configured": _is_configured(),
        "setup_complete": _setup_complete,
        "asset_id": ITWIN_ASSET_ID if ITWIN_ENABLED else None,
        "sensor_ids": {k: v for k, v in _sensor_ids.items()},
        "last_push": _last_push_status,
        "last_pushed_data_ts": _last_pushed_data_ts,
        "token_valid": (
            _token_cache["access_token"] is not None
            and time.time() < _token_cache["expires_at"]
        ),
    }
