"""Distribution-shift status and machine alert level (CAEE revision, decisions B1 and B2, 14 Sep 2026).

B1 - status per channel. s-IDK^2 [Ting et al., VLDB J. 33 (2024) 753-780] returns a similarity score per
sliding window and ranks windows by it; the paper defines no alarm threshold. This module adds the decision
rule used by the system: a window is flagged when its similarity falls below

        median(S) - K * 1.4826 * MAD(S),

where S are the scores of the current population (the last 3 days) of that channel (Hampel identifier, K = 3).
The rule needs no labels and can flag no window at all, in line with the paper's definition of anomalies as
rare. Because a persistent change becomes the majority of the population after about half of it, its scores
rise again; the first detection is therefore kept as an open shift episode until an operator reviews it.

B2 - alert level. The ISO 10816 zone of z_rms, the shift status and the 6 h forecast are fused by a fixed
rule. Zones are read relative to the baseline zone, the zone of established operation at commissioning (BASELINE_ZONE, "C" for this
conveyor), because this conveyor runs in zone C when healthy:

    zone D                                   -> High
    zone worse than baseline (not D)         -> High if shift or forecast worsens, else Medium
    zone at or better than baseline          -> Medium if shift and forecast worsens,
                                                Low if shift or forecast worsens, else Normal

"forecast worsens" means the 6 h forecast of z_rms lies in a worse zone than the current reading and the
increase exceeds the recent 6 h forecast MAE (when that error is available).
"""
import os

import numpy as np
import pandas as pd

K_HAMPEL = float(os.getenv("SHIFT_RULE_K", "3"))
# Number of consecutive 30-minute cycles a channel must stay flagged before it counts as a shift.
# 1 = flag at once (approved rule). Larger values trade detection delay for fewer flags (E16b replay).
PERSISTENCE_CYCLES = max(1, int(os.getenv("SHIFT_RULE_PERSISTENCE", "1")))
MAD_SCALE = 1.4826
BASELINE_ZONE = os.getenv("ISO_BASELINE_ZONE", "C").upper()
SHIFT_CHANNELS = ["z_rms", "x_rms", "z_peak", "x_peak", "noise"]  # vibration and noise channels used for the alert
EVENT_DAYS = 7
# An episode left unreviewed for this many hours of data raises a reminder for the operator (one work shift).
# The reminder is a notice only: it does not start the agent or draft a work order.
REMINDER_HOURS = float(os.getenv("SHIFT_REMINDER_HOURS", "8"))
ZONES = ["A", "B", "C", "D"]
ZONE_LIMITS = [(0.71, "A"), (1.8, "B"), (4.5, "C")]  # ISO 10816-1 class I, mm/s RMS
ZONE_NAMES = {"A": "Good", "B": "Acceptable", "C": "Unsatisfactory", "D": "Unacceptable"}


def iso_zone(z_rms: float) -> str:
    for limit, zone in ZONE_LIMITS:
        if z_rms < limit:
            return zone
    return "D"


def shift_threshold(scores) -> float:
    s = np.asarray(scores, dtype=float)
    s = s[np.isfinite(s)]
    if s.size == 0:
        return float("nan")
    med = float(np.median(s))
    mad = float(np.median(np.abs(s - med)))
    return med - K_HAMPEL * MAD_SCALE * mad


def channel_status(scores, timestamps) -> dict:
    """Threshold, flags and latest status of one channel's window scores (aligned to window-end timestamps)."""
    s = np.asarray(scores, dtype=float).ravel()
    thr = shift_threshold(s)
    flags = s < thr if np.isfinite(thr) else np.zeros(len(s), dtype=bool)
    return {
        "threshold": round(thr, 4) if np.isfinite(thr) else None,
        "latest_score": round(float(s[-1]), 4) if len(s) else None,
        "latest_flag": bool(flags[-1]) if len(s) else False,
        "flagged": [(pd.Timestamp(t), round(float(v), 4)) for t, v, f in zip(timestamps, s, flags) if f],
    }


def update_events(events: list, statuses: dict, now) -> list:
    """Merge newly flagged windows into the rolling event log (deduplicated by time and channel)."""
    known = {(e["timestamp"], e["sensor"]) for e in events}
    for ch, st in statuses.items():
        for ts, score in st["flagged"]:
            key = (str(ts), ch)
            if key not in known:
                events.append({"timestamp": str(ts), "sensor": ch, "score": score, "threshold": st["threshold"]})
                known.add(key)
    cutoff = pd.Timestamp(now) - pd.Timedelta(days=EVENT_DAYS)
    events = [e for e in events if pd.Timestamp(e["timestamp"]) >= cutoff]
    events.sort(key=lambda e: (e["timestamp"], e["sensor"]))
    return events


def update_episode(episode, statuses: dict, now):
    """Open an episode at the first flagged window of a shift channel; keep it open until reviewed."""
    active = sorted(ch for ch in SHIFT_CHANNELS if ch in statuses and statuses[ch]["latest_flag"])
    if episode is None or episode.get("reviewed"):
        if not active:
            return episode if episode and not episode.get("reviewed") else None
        return {"since": str(pd.Timestamp(now)), "channels": active, "currently_flagged": active,
                "reviewed": False, "reviewed_by": None}
    episode = dict(episode)
    episode["channels"] = sorted(set(episode["channels"]) | set(active))
    episode["currently_flagged"] = active
    return episode


def reminder(episode, now, hours: float = None):
    """Reminder when a shift episode has stayed open without review for REMINDER_HOURS of data, else None."""
    limit = REMINDER_HOURS if hours is None else hours
    if not episode or episode.get("reviewed"):
        return None
    open_h = (pd.Timestamp(now) - pd.Timestamp(episode["since"])).total_seconds() / 3600
    if open_h < limit:
        return None
    flagged_now = episode.get("currently_flagged") or []
    return {
        "since": episode["since"], "open_hours": round(open_h, 1), "limit_hours": limit,
        "channels": episode["channels"], "currently_flagged": flagged_now,
        "message": (f"Distribution-shift episode open for {open_h:.1f} h without review (since {episode['since']}; "
                    f"channels: {', '.join(episode['channels'])}; "
                    + (f"still flagged: {', '.join(flagged_now)}" if flagged_now else "none flagged in the latest window")
                    + "). Check the trend and mark the episode as reviewed, or ask the copilot for a work order if "
                      "an intervention is needed."),
    }


def forecast_worsens(z_now: float, z_forecast_6h, mae_6h) -> bool:
    if z_forecast_6h is None:
        return False
    worse_zone = ZONES.index(iso_zone(z_forecast_6h)) > ZONES.index(iso_zone(z_now))
    margin = mae_6h if mae_6h is not None else 0.0
    return bool(worse_zone and (z_forecast_6h - z_now) > margin)


def alert_level(zone: str, shift: bool, forecast_worse: bool, baseline: str = BASELINE_ZONE) -> str:
    if zone == "D":
        return "High"
    if ZONES.index(zone) > ZONES.index(baseline):
        return "High" if (shift or forecast_worse) else "Medium"
    if shift and forecast_worse:
        return "Medium"
    if shift or forecast_worse:
        return "Low"
    return "Normal"


def work_order_scope(zone: str, level: str, baseline: str = BASELINE_ZONE) -> str:
    """Scope of a work order drafted at the operator's request (fixed rule, decided by the authors on 15 Sep 2026).

    monitoring            z_rms in a better zone than the baseline zone, or in that zone at level Normal:
                          no cause analysis and no inspection, only routine monitoring
    scheduled_inspection  baseline zone (or a worse zone below High) at level Low or Medium:
                          possible causes from the manual and checks at the next scheduled maintenance
    prompt_inspection     level High: possible causes and checks as soon as practical
    Noise and temperature also bear on safety, but no standard defines severity zones for them as
    ISO 10816 does for vibration, so the rule and the scope are set by vibration only.
    """
    if level == "High":
        return "prompt_inspection"
    if ZONES.index(zone) < ZONES.index(baseline) or level == "Normal":
        return "monitoring"
    return "scheduled_inspection"


def assess(z_now: float, episode, z_forecast_6h=None, mae_6h=None, baseline: str = BASELINE_ZONE) -> dict:
    """Alert level with the reasons the agent reports to the operator."""
    zone = iso_zone(z_now)
    shift = bool(episode) and not episode.get("reviewed")
    worse = forecast_worsens(z_now, z_forecast_6h, mae_6h)
    level = alert_level(zone, shift, worse, baseline)
    reasons = [f"ISO 10816 class I zone {zone} ({ZONE_NAMES[zone]}) at z_rms = {z_now:.2f} mm/s; "
               f"zone {baseline} is the baseline zone of this conveyor"]
    if shift:
        flagged_now = episode.get("currently_flagged") or []
        reasons.append(f"distribution-shift episode open since {episode['since']}, not yet reviewed by an operator; "
                       f"channels flagged during the episode: {', '.join(episode['channels'])}"
                       + (f"; flagged in the latest window: {', '.join(flagged_now)}" if flagged_now
                          else "; none flagged in the latest window"))
    else:
        reasons.append("no unreviewed distribution shift")
    if z_forecast_6h is not None:
        reasons.append(f"z_rms forecast in 6 h = {z_forecast_6h:.2f} mm/s (zone {iso_zone(z_forecast_6h)})"
                       + ("; worse than now beyond the recent forecast error" if worse else ""))
    return {"level": level, "zone": zone, "shift": shift, "forecast_worsens": worse,
            "baseline_zone": baseline, "reasons": reasons, "work_order_scope": work_order_scope(zone, level, baseline)}
