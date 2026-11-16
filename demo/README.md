# Demo: the 5 May 2026 incident (FOI 2026 video)

The demo video shows the copilot on the conveyor as it was on **5 May 2026 at 15:00**, a few hours after a load
was added: z_rms 3.54 mm/s (ISO 10816 zone C), a distribution-shift episode open since 08:30, alert level Low.
The live dashboard shows today's data, so its answers differ from the video. This folder replays that afternoon
on a local machine, with the same backend code, the same machine state and the same work-order history.

| File | Purpose |
|---|---|
| `replay_may2026.py` | Serves the dashboard API on the replayed state (http://127.0.0.1:8000) |
| `check_video.py` | Plays the video's questions without the browser and compares the answers with the video |
| `seed_work_orders.py` | Adds the two seed work orders to another persistent store, e.g. the local backend's |
| `../backend/seed_work_orders.json` | The two earlier work orders shown in the video (benchmark fixtures WO-E and WO-C); the demo deployment adds them at start-up |
| `data/conveyor_export.csv` | Recorded sensor data (from the sufficiency-gated-maintenance-agent repository) |

## Setup (once)
1. `backend/.env` with `OPENAI_API_KEY=...` (the replay needs no MySQL).
2. Python packages: `pip install -r backend/requirements.txt`; frontend: `cd frontend && npm install`.
3. First start: `python demo/replay_may2026.py` trains the forecast models for the replayed cycles
   (about 5 minutes); they are cached in `demo/cache/`, later starts take about 30 s.

## Running the demo
```bash
python demo/replay_may2026.py        # terminal 1: wait for "Uvicorn running on http://127.0.0.1:8000"
cd frontend && npm run dev:local     # terminal 2: open http://localhost:5173
```
The header shows "replay of recorded data up to 05 May 2026 15:00". Every start of the replay begins with the
two seed work orders only, so restart it to rehearse again.

Script of the video (open the copilot, type each question exactly):

1. `Why is the alert level Low right now?`: answered from the machine state, no tool.
2. `Search our past work orders: is there a work order for this vibration and noise change?`:
   the gate finds no close record (distance about 1.49) and the agent abstains.
3. `Please draft a work order for the current machine condition.`: manual search, gate (insufficient),
   draft with priority Low for a scheduled inspection.
4. In the draft panel, replace the first sentence of the Root Cause Analysis ("No matching past work order
   exists.") with `Rise in vibration and noise after the added load on 5 May 2026, reviewed by the operator.`
   and click **Approve & Save**. The record appears in the Work Order History; the alert level becomes Normal
   (the approval also reviews the shift episode).
5. Click **New Chat** and ask question 2 again: the gate now finds the approved record sufficient (distance
   about 0.91) and the agent answers from it.

## Checking the answers before the demo
```bash
python demo/check_video.py           # about 2 minutes; report in demo/results/check_<time>.md
```
The check confirms what must match the video (tools, gate verdicts, priority, alert level after approval, records)
and stores the full answers. The wording is the model's own and can differ slightly between runs, even at
temperature 0: the replay runs the copilot at temperature 0 with a fixed seed (`REPLAY_DETERMINISTIC=0` restores
the live 0.1), which makes rehearsal and demo close, but OpenAI does not guarantee identical output.
The most visible variation is in step 3: the cause list follows the manual passages the agent's own search
returns ("contamination" in the video, sometimes "worn bearings in the drive unit"); both come from the manual's
troubleshooting table.
