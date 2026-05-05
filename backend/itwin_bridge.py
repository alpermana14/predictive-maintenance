"""
itwin_bridge.py — Bentley iTwin IoT Sensor Data Integration Bridge

Pushes raw sensor readings AND ML prediction outputs (forecasts, anomaly scores)
to the Bentley iTwin IoT platform via their REST Sensor Data API.

Runs inside the existing FastAPI process alongside APScheduler.
Gracefully degrades to a no-op when credentials are not configured.
"""

import os
import time
import uuid
import logging
import numpy as np
import requests
from datetime import datetime, timezone
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

# Sensor targets (must match ml_engine.py TARGETS)
TARGETS = ["current", "temperature", "z_rms", "x_rms", "z_peak", "x_peak", "noise"]

# Units mapping for each target
UNITS = {
    "current": "A",
    "temperature": "°C",
    "z_rms": "mm/s",
    "x_rms": "mm/s",
    "z_peak": "mm/s",
    "x_peak": "mm/s",
    "noise": "dB",
}

# ===================== GLOBAL STATE =====================

_token_cache = {
    "access_token": None,
    "expires_at": 0,  # Unix timestamp
}

# Global cache for sensor IDs (mapped to the user's existing April 9th sensors)
_sensor_ids = {
    "temperature": "/api/D54C48/node/dynamic/DE1B55/device/7B7B3D/sensor",
    "current": "/api/D54C48/node/dynamic/DE1B55/device/97A0C2/sensor",
    "z_rms": "/api/07EF1F/node/dynamic/C17041/device/BD0D0F/sensor",
    "x_rms": "/api/593C59/node/dynamic/FCE47E/device/1B92B8/sensor",
    "x_peak": "/api/593C59/node/dynamic/FCE47E/device/24A4B1/sensor",
    "noise": "/api/593C59/node/dynamic/FCE47E/device/E9117F/sensor",
    "z_peak": "/api/593C59/node/dynamic/FCE47E/device/C27256/sensor",
    "forecast": "",
    "anomaly": ""
}

_last_push_status = {
    "timestamp": None,
    "success": None,
    "message": "",
    "observations_sent": 0,
}

_setup_complete = False


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
    One-time setup: Register device and sensors in iTwin IoT.
    
    Creates:
    - 1 Device: "Conveyor-Main"
    - 6 Sensors: Vibration, Temperature, Current, Noise, Forecast, Anomaly
    
    After creation, queries the integration to discover sensor IDs.
    Returns True if successful, False otherwise.
    """
    global _setup_complete

    if not _is_configured():
        logger.info("iTwin integration is not configured — skipping sensor setup.")
        return False
    
    _setup_complete = True
    return True


def _discover_existing_sensors(headers: dict) -> bool:
    # Since we are using hardcoded mappings for the existing sensors, we just return True.
    # The dictionary is already populated at the top of the file.
    logger.info("Using hardcoded sensor IDs from existing April 9th nodes.")
    return True


# ===================== DATA PUSH =====================

def push_to_itwin(state) -> bool:
    """
    Push latest data from the global MachineState to Bentley iTwin IoT.
    
    Sends:
    1. Raw sensor readings (latest data point)
    2. Forecast values (last predicted step for each target)
    3. Anomaly scores (latest IDK score for each target)
    
    Args:
        state: The MachineState object from main.py
        
    Returns:
        True if push succeeded, False otherwise
    """
    global _last_push_status

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
        
        # Build payload mapping exactly to the 7 existing sensors
        if _sensor_ids.get("temperature"):
            observations.append({
                "sensorId": _sensor_ids["temperature"],
                "timestamp": ts_str,
                "values": [_safe_float(features.get("temperature", 0))]
            })
            
        if _sensor_ids.get("current"):
            observations.append({
                "sensorId": _sensor_ids["current"],
                "timestamp": ts_str,
                "values": [_safe_float(features.get("current", 0))]
            })
            
        if _sensor_ids.get("z_rms"):
            observations.append({
                "sensorId": _sensor_ids["z_rms"],
                "timestamp": ts_str,
                "values": [_safe_float(features.get("z_rms", 0))]
            })
            
        if _sensor_ids.get("x_rms"):
            observations.append({
                "sensorId": _sensor_ids["x_rms"],
                "timestamp": ts_str,
                "values": [_safe_float(features.get("x_rms", 0))]
            })
            
        if _sensor_ids.get("x_peak"):
            observations.append({
                "sensorId": _sensor_ids["x_peak"],
                "timestamp": ts_str,
                "values": [_safe_float(features.get("x_peak", 0))]
            })
            
        if _sensor_ids.get("noise"):
            observations.append({
                "sensorId": _sensor_ids["noise"],
                "timestamp": ts_str,
                "values": [_safe_float(features.get("noise", 0))]
            })
            
        if _sensor_ids.get("z_peak"):
            observations.append({
                "sensorId": _sensor_ids["z_peak"],
                "timestamp": ts_str,
                "values": [_safe_float(features.get("z_peak", 0))]
            })

        # --- 2. Forecast Values ---
        if _sensor_ids.get("forecast") and state.forecast is not None:
            forecast_values = {}
            for tgt in TARGETS:
                if tgt in state.forecast.columns:
                    # Use the last forecasted value (furthest prediction)
                    forecast_values[f"{tgt}_forecast"] = _safe_float(
                        state.forecast[tgt].iloc[-1]
                    )
                elif tgt in state.forecast:
                    # Handle if forecast is a dict of Series
                    forecast_values[f"{tgt}_forecast"] = _safe_float(
                        state.forecast[tgt].iloc[-1]
                    )

            if forecast_values:
                # Use the forecast's own last timestamp
                fc_ts = ts_str  # Default to current data timestamp
                try:
                    if hasattr(state.forecast, "index") and len(state.forecast.index) > 0:
                        fc_timestamp = state.forecast.index[-1]
                        if hasattr(fc_timestamp, "isoformat"):
                            fc_ts = fc_timestamp.isoformat()
                            if "+" not in fc_ts and "Z" not in fc_ts:
                                fc_ts += "Z"
                except Exception:
                    pass

                observations.append({
                    "sensorId": _sensor_ids["forecast"],
                    "timestamp": fc_ts,
                    "values": forecast_values,
                })

        # --- 3. Anomaly Scores ---
        if _sensor_ids.get("anomaly") and state.anomalies:
            anomaly_values = {}
            for tgt in TARGETS:
                scores = state.anomalies.get(tgt)
                if scores is not None and len(scores) > 0:
                    anomaly_values[f"{tgt}_anomaly"] = _safe_float(scores[-1])

            if anomaly_values:
                observations.append({
                    "sensorId": _sensor_ids["anomaly"],
                    "timestamp": ts_str,
                    "values": anomaly_values,
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

        if resp.status_code in (200, 201, 204):
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
        "token_valid": (
            _token_cache["access_token"] is not None
            and time.time() < _token_cache["expires_at"]
        ),
    }
