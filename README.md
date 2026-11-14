# Predictive Maintenance Copilot: Agentic AI & ML Forecasting

This project combines **Traditional Machine Learning** for time-series forecasting with **Agentic AI** to provide a "Copilot" experience for industrial maintenance. It predicts sensor failures on a conveyor belt and allows technicians to interact with the data through a natural language interface.

---

### 🚀 Features
* **Time-Series Forecasting:** Uses **LightGBM** to predict future trends for vibration (RMS), temperature, and current.
* **Anomaly Detection:** Implements a custom **IDK (Isolation Distribution Kernel)** algorithm to detect sensor irregularities.
* **Agentic AI:** A **LangGraph-powered** agent that can query machine manuals (PDF), search maintenance history, and draft work orders.
* **Real-time Scheduling:** An automated background scheduler that updates the machine state every 5 minutes from a MySQL database.

---

### 🛠️ Tech Stack
* **Backend:** Python (FastAPI), LangChain, LangGraph, LightGBM, Pandas, ChromaDB.
* **Frontend:** React.
* **Database:** MySQL (Sensor data) & Chroma (Vector storage).

---

### 📋 Prerequisites
Before running the project, ensure you have:
* **Python 3.10+**
* **Node.js** (for React frontend)
* **An OpenAI API Key**

---

### 🔧 Installation & Setup

#### 1. Backend Setup
```bash
cd backend
python -m venv venv
.\venv\Scripts\Activate.ps1  # Windows
pip install -r requirements.txt
```

#### 2. Frontend Setup
```bash
cd frontend
npm install
npm run dev
```

### 🧠 How the Agent Works
The AI Copilot uses a Dynamic System Prompt that receives live machine context (current vibration, ISO 10816 zones, and forecast trends). It can execute three primary tools:

* **Manual Retriever:** Searches the Maintenance_Conveyor.pdf for technical specifications.
* **History Search:** Queries past maintenance logs stored in ChromaDB.
* **Work Order Drafter:** Automatically populates maintenance drafts based on detected anomalies.

### 🏭 Digital Twin
The dashboard includes a 3D digital twin of the lab that houses the conveyor. It is built from a laser scan through a Scan-to-BIM model in Revit and is served entirely by this frontend: no Autodesk Tandem, Bentley iTwin or other third-party viewer or CDN is involved, so the model and the scan stay on our own server.

#### How it was built
```
Laser scan (E57 + LAS) ──► point-cloud processing (Python) ──► Revit model (Dynamo scripts)
                                      │                                  │
                                      └──────── JSON model sources ──────┘
                                                      │
                                     export_twin.py / export_cloud.py
                                                      │
                          frontend/public/twin/twin.json + cloud.bin ──► three.js viewer ◄── /api/summary
```
1. **Scan.** The room was captured with a 4D Meta laser scanner: an E57 file with about 3.7 million coloured points and two 360° panoramas, plus a LAS export.
2. **Point-cloud processing (Python).** A small E57 reader extracts the points and colours. A RANSAC plane fit on the floor and ceiling corrects a 1.95° tilt and puts the scan in a room frame (x east, y north, z up, metres, origin at the south-west corner). From this frame the scripts measure the room outline and ceiling height (2.97 m), the furniture, the conveyor and the motor, the windows, doors, lights and signage, and the colour of every surface (the median RGB of the scan points on it).
3. **Point cloud for Revit.** The levelled scan is converted to a ReCap project (`.rcp`) with ReCap Pro and linked into Revit at the internal origin, so the model can be checked against the scan.
4. **Revit model (Dynamo).** Three Dynamo graphs (Python, Revit API) build the model from JSON files:
   * stage 1 (`geometry.json`): walls, floor, ceiling, door, room and the point-cloud link;
   * stage 2 (`furniture.json`): desks, chairs, display, AC, the crescent-plate conveyor `CONV-01` and its drive motor `MTR-01`, as DirectShapes with scan colours;
   * stage 3 (`scene.json`): paint from the scan, window openings, doors, ceiling grid, lights, diffusers, projector, speakers, conduits, sockets, the lettering and the UMPSA logo (vectorised from the official logo file with OpenCV).

   Every element carries a Mark (for example `CONV-01`), which is also its ID in the web model.
5. **Web export.** `export_twin.py` runs the same geometry builders as the Revit scripts and writes `twin.json`: every element as extruded 2D profiles with colours, plus the sensor positions. `export_cloud.py` writes `cloud.bin`: 600k scan points quantised to 16 bits per axis and grouped into interior, ceiling and per-wall segments.
6. **Viewer.** `DigitalTwin.jsx` turns the profiles into three.js meshes and shows them next to the live sensor data. Walls, the ceiling and the matching point-cloud segments between the camera and the room are hidden automatically (cutaway), so the inside of the room stays visible from any angle.

#### What the viewer shows
* **Model:** `frontend/public/twin/twin.json` (about 95 KB) with walls, doors, windows, ceiling, lights, furniture, the conveyor (`CONV-01`) and its drive motor (`MTR-01`).
* **Point cloud:** `frontend/public/twin/cloud.bin` (5.4 MB), loaded only when the *Point cloud* button is pressed.
* **Live data:** the viewer reuses `/api/summary`, which the dashboard polls every 5 s. The box around the conveyor takes the colour of the ISO 10816 zone, channels flagged by the distribution-shift episode are marked with ⚠, and clicking a sensor value selects it in the forecast and diagnostics charts.
* **Sensors** (vibration, temperature, noise) are mounted on the drive motor under the conveyor. Their positions are listed under `sensors` in `twin.json`.

#### Technologies
| Layer | Technology | Licence |
|---|---|---|
| Scan | 4D Meta laser scanner (E57, LAS) | – |
| Point-cloud processing | Python, NumPy, OpenCV, Matplotlib, Pillow | BSD / Apache-2.0 / PSF / HPND |
| Point cloud in Revit | Autodesk ReCap Pro 2027 | commercial (student licence) |
| BIM model | Autodesk Revit 2027, Dynamo 4.2 (PythonNet3), Revit API | commercial (education licence) |
| 3D viewer | three.js (ExtrudeGeometry, OrbitControls, CSS2DRenderer) | MIT |
| Frontend | React 19, Vite, Tailwind CSS; the viewer is a separate lazy-loaded chunk | MIT |
| Hosting | nginx in Docker on CapRover, deployed by GitHub Actions | BSD-2 / Apache-2.0 |

The Revit model and its JSON sources are kept in the Scan-to-BIM workspace, outside this repository. After the model changes, regenerate the web files there:
```bash
python export_twin.py  <repo>/frontend/public/twin
python export_cloud.py <repo>/frontend/public/twin
```

**Limitations.** The model is at about LOD 200: furniture, the conveyor and the accessories are simplified shapes measured from the scan, the sensor positions on the motor are estimates, and the point cloud also contains the people who were in the room during the scan.
