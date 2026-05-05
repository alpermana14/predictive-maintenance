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

# Discovered sensor IDs after registration (populated by setup_itwin_sensors)
_sensor_ids = {
    "vibration": None,
    "temperature": None,
    "current": None,
    "noise": None,
    "forecast": None,
    "anomaly": None,
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

    headers = _auth_headers()
    if not headers:
        logger.error("Cannot setup sensors — failed to obtain access token.")
        return False

    # First, try to discover existing sensors
    if _discover_existing_sensors(headers):
        logger.info("Existing sensors discovered — skipping registration.")
        _setup_complete = True
        return True

    # Register new device + sensors
    logger.info("Registering device and sensors in iTwin IoT...")

    payload = {
        "integration": {
            "changeState": "new",
            "devices": [
                {
                    "changeState": "new",
                    "refId": str(uuid.uuid4()),
                    "props": {
                        "INTEGRATION_ID": "IMPORT_DEVICE_SDE",
                        "NAME": "Conveyor-Main",
                    },
                    "sensors": [
                        # Sensor 1: Vibration (4 metrics)
                        {
                            "changeState": "new",
                            "refId": str(uuid.uuid4()),
                            "props": {
                                "INTEGRATION_ID": "GENERIC_SENSOR_SDE",
                                "NAME": "Vibration-Sensor",
                                "UNKNOWN_UNITS": {"0": "mm/s", "1": "mm/s", "2": "mm/s", "3": "mm/s"},
                                "UNKNOWN_METRICS": {"0": "z_rms", "1": "x_rms", "2": "z_peak", "3": "x_peak"},
                            },
                        },
                        # Sensor 2: Temperature
                        {
                            "changeState": "new",
                            "refId": str(uuid.uuid4()),
                            "props": {
                                "INTEGRATION_ID": "GENERIC_SENSOR_SDE",
                                "NAME": "Temperature-Sensor",
                                "UNKNOWN_UNITS": {"0": "degC"},
                                "UNKNOWN_METRICS": {"0": "temperature"},
                            },
                        },
                        # Sensor 3: Current
                        {
                            "changeState": "new",
                            "refId": str(uuid.uuid4()),
                            "props": {
                                "INTEGRATION_ID": "GENERIC_SENSOR_SDE",
                                "NAME": "Current-Sensor",
                                "UNKNOWN_UNITS": {"0": "A"},
                                "UNKNOWN_METRICS": {"0": "current"},
                            },
                        },
                        # Sensor 4: Noise
                        {
                            "changeState": "new",
                            "refId": str(uuid.uuid4()),
                            "props": {
                                "INTEGRATION_ID": "GENERIC_SENSOR_SDE",
                                "NAME": "Noise-Sensor",
                                "UNKNOWN_UNITS": {"0": "dB"},
                                "UNKNOWN_METRICS": {"0": "noise"},
                            },
                        },
                        # Sensor 5: Forecast (all targets combined)
                        {
                            "changeState": "new",
                            "refId": str(uuid.uuid4()),
                            "props": {
                                "INTEGRATION_ID": "GENERIC_SENSOR_SDE",
                                "NAME": "Forecast-Sensor",
                                "UNKNOWN_UNITS": {
                                    "0": "A", "1": "degC", "2": "mm/s",
                                    "3": "mm/s", "4": "mm/s", "5": "mm/s", "6": "dB",
                                },
                                "UNKNOWN_METRICS": {
                                    "0": "current_forecast",
                                    "1": "temperature_forecast",
                                    "2": "z_rms_forecast",
                                    "3": "x_rms_forecast",
                                    "4": "z_peak_forecast",
                                    "5": "x_peak_forecast",
                                    "6": "noise_forecast",
                                },
                            },
                        },
                        # Sensor 6: Anomaly Scores
                        {
                            "changeState": "new",
                            "refId": str(uuid.uuid4()),
                            "props": {
                                "INTEGRATION_ID": "GENERIC_SENSOR_SDE",
                                "NAME": "Anomaly-Sensor",
                                "UNKNOWN_UNITS": {
                                    "0": "score", "1": "score", "2": "score",
                                    "3": "score", "4": "score", "5": "score", "6": "score",
                                },
                                "UNKNOWN_METRICS": {
                                    "0": "current_anomaly",
                                    "1": "temperature_anomaly",
                                    "2": "z_rms_anomaly",
                                    "3": "x_rms_anomaly",
                                    "4": "z_peak_anomaly",
                                    "5": "x_peak_anomaly",
                                    "6": "noise_anomaly",
                                },
                            },
                        },
                    ],
                }
            ],
        }
    }

    try:
        url = f"{INTEGRATE_URL}?iTwinId={ITWIN_ASSET_ID}"
        logger.info(f"POST {url}")
        resp = requests.post(url, json=payload, headers=headers, timeout=60)

        logger.info(f"Registration response status: {resp.status_code}")
        logger.info(f"Registration response body: {resp.text[:1000]}")

        if resp.status_code in (200, 201):
            logger.info(f"Sensor registration successful ({resp.status_code})")
            # Now discover the created sensor IDs
            _discover_existing_sensors(headers)
            _setup_complete = True
            return True
        else:
            logger.error(f"Sensor registration failed ({resp.status_code}): {resp.text[:1000]}")
            return False

    except Exception as e:
        logger.error(f"Sensor registration exception: {e}")
        return False


def _discover_existing_sensors(headers: dict) -> bool:
    """
    Query the integration to find existing sensor IDs.
    Populates the _sensor_ids dict with discovered paths.
    Returns True if sensors were found.
    """
    try:
        # Try to use the GET nodes endpoint to find existing integrations
        url = f"https://api.bentley.com/sensor-data/integrations/nodes?iTwinId={ITWIN_ASSET_ID}"
        resp = requests.get(
            url,
            headers=headers,
            timeout=30,
        )

        logger.info(f"Discovery response status: {resp.status_code}")

        if resp.status_code != 200:
            logger.warning(f"Discovery query returned {resp.status_code}: {resp.text[:500]}")
            return False

        data = resp.json()
        logger.info(f"Discovery response body: {str(data)[:1000]}")

        # If data contains a list of nodes, we need to find our device and sensors
        # Note: the exact structure depends on Bentley's GET nodes response. 
        # We will iterate through all elements recursively to find "sensors" with our target names
        
        found_count = 0
        
        def _search_sensors(obj):
            nonlocal found_count
            if isinstance(obj, dict):
                if "sensors" in obj and isinstance(obj["sensors"], list):
                    for sensor in obj["sensors"]:
                        sensor_name = sensor.get("props", {}).get("NAME", "")
                        sensor_id = sensor.get("id", "")
                        
                        name_to_key = {
                            "Vibration-Sensor": "vibration",
                            "Temperature-Sensor": "temperature",
                            "Current-Sensor": "current",
                            "Noise-Sensor": "noise",
                            "Forecast-Sensor": "forecast",
                            "Anomaly-Sensor": "anomaly",
                        }
                        
                        key = name_to_key.get(sensor_name)
                        if key and sensor_id:
                            _sensor_ids[key] = sensor_id
                            found_count += 1
                            logger.info(f"Discovered sensor: {sensor_name} → {sensor_id}")
                
                for k, v in obj.items():
                    _search_sensors(v)
            elif isinstance(obj, list):
                for item in obj:
                    _search_sensors(item)

        _search_sensors(data)

        # Parse the response to find sensor paths by name
        # The response structure varies; we look for sensors with our known names
        integration = data.get("integration", data)
        devices = integration.get("devices", [])

        found_count = 0
        for device in devices:
            device_name = device.get("props", {}).get("NAME", "")
            if device_name != "Conveyor-Main":
                continue

            sensors = device.get("sensors", [])
            for sensor in sensors:
                sensor_name = sensor.get("props", {}).get("NAME", "")
                sensor_id = sensor.get("id", "")

                name_to_key = {
                    "Vibration-Sensor": "vibration",
                    "Temperature-Sensor": "temperature",
                    "Current-Sensor": "current",
                    "Noise-Sensor": "noise",
                    "Forecast-Sensor": "forecast",
                    "Anomaly-Sensor": "anomaly",
                }

                key = name_to_key.get(sensor_name)
                if key and sensor_id:
                    _sensor_ids[key] = sensor_id
                    found_count += 1
                    logger.info(f"Discovered sensor: {sensor_name} → {sensor_id}")

        if found_count > 0:
            logger.info(f"Discovered {found_count} existing sensors.")
            
            # Save to a local file cache for faster recovery
            try:
                import json
                with open("itwin_sensor_ids.json", "w") as f:
                    json.dump(_sensor_ids, f)
            except Exception:
                pass
                
            return True

        # Also try to load from local cache if API didn't find them
        try:
            import json
            if os.path.exists("itwin_sensor_ids.json"):
                with open("itwin_sensor_ids.json", "r") as f:
                    cached_ids = json.load(f)
                    for k, v in cached_ids.items():
                        if v:
                            _sensor_ids[k] = v
                            found_count += 1
                if found_count > 0:
                    logger.info(f"Loaded {found_count} existing sensors from local cache.")
                    return True
        except Exception:
            pass

        return False

    except Exception as e:
        logger.debug(f"Sensor discovery exception: {e}")
        return False


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
        observations = []
        latest = state.data.iloc[-1]
        # Use the sensor data timestamp in UTC ISO format
        timestamp = latest.name
        if hasattr(timestamp, "isoformat"):
            ts_str = timestamp.isoformat()
            # Ensure UTC format for Bentley
            if "+" not in ts_str and "Z" not in ts_str:
                ts_str += "Z"
        else:
            ts_str = str(timestamp)

        # --- 1. Raw Sensor Readings ---

        # Vibration sensor (4 metrics)
        if _sensor_ids.get("vibration"):
            observations.append({
                "sensorId": _sensor_ids["vibration"],
                "timestamp": ts_str,
                "values": {
                    "z_rms": _safe_float(latest.get("z_rms")),
                    "x_rms": _safe_float(latest.get("x_rms")),
                    "z_peak": _safe_float(latest.get("z_peak")),
                    "x_peak": _safe_float(latest.get("x_peak")),
                },
            })

        # Temperature sensor
        if _sensor_ids.get("temperature"):
            observations.append({
                "sensorId": _sensor_ids["temperature"],
                "timestamp": ts_str,
                "values": {
                    "temperature": _safe_float(latest.get("temperature")),
                },
            })

        # Current sensor
        if _sensor_ids.get("current"):
            observations.append({
                "sensorId": _sensor_ids["current"],
                "timestamp": ts_str,
                "values": {
                    "current": _safe_float(latest.get("current")),
                },
            })

        # Noise sensor
        if _sensor_ids.get("noise"):
            observations.append({
                "sensorId": _sensor_ids["noise"],
                "timestamp": ts_str,
                "values": {
                    "noise": _safe_float(latest.get("noise")),
                },
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
        r1 = requests.get(f"https://api.bentley.com/sensor-data/integrations/nodes?iTwinId={ITWIN_ASSET_ID}", headers=headers)
        result["nodes"] = r1.json() if r1.status_code == 200 else r1.text
        
        # Get Devices
        r2 = requests.get(f"https://api.bentley.com/sensor-data/devices?iTwinId={ITWIN_ASSET_ID}", headers=headers)
        result["devices"] = r2.json() if r2.status_code == 200 else r2.text
        
        # Get Sensors
        r3 = requests.get(f"https://api.bentley.com/sensor-data/sensors?iTwinId={ITWIN_ASSET_ID}", headers=headers)
        result["sensors"] = r3.json() if r3.status_code == 200 else r3.text
        
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
