// Mixed-reality view of the digital twin for Meta Quest 3 (WebXR 'immersive-ar' in the Meta Quest Browser).
//
// Three modes:
//   onsite   : the twin at 1:1 in the real lab. It is aligned automatically by fitting the room outline to the walls
//              of the headset's Space Setup (WebXR plane detection); two floor corners aimed at with the ray are the
//              fallback. The alignment is kept as a persistent WebXR anchor (and in localStorage), so it is restored
//              in every later session on this headset and account.
//   conveyor : only the conveyor at 1:1, aligned by touching two corners of its frame with the controller (no ray).
//              It shares the saved alignment with onsite, so aligning here also corrects the on-site view at the
//              conveyor.
//   tabletop : a scale model of the lab placed on a table or the floor.
//
// The live machine state comes from the same /api/summary object the dashboard already polls (setSummary).
// Human in the loop: the right GRIP opens an actions menu next to the status card (acknowledge the shift episode,
// approve or reject a copilot draft, ask the copilot preset questions). The actions use the dashboard's endpoints and
// approver token (hitl.js), so the headset changes the maintenance record only, never the machine.
// Measurement sessions (?measure in the URL): the left GRIP toggles registration checkpoints, and alignment results,
// frame-rate samples and action timings are posted to /api/ar/log.
// In-headset UI is drawn on canvas textures, because DOM overlays are not available in immersive mode on Quest.
import * as THREE from 'three';
import { ZONE_COLORS, NEUTRAL, FLAGGED, toThree, buildMaterials, buildElement, loadCloud, wallPlanes, sensorColor, fmt } from './twinModel';
import { approverToken, timedPost, fetchSummary, logRecord, flushLog, MEASURE } from './hitl';

const STORAGE_KEY = 'pm-twin-ar-calibration-v1';
const CLOUD_POINT_SIZE = 0.012;
const Y_AXIS = new THREE.Vector3(0, 1, 0);

// Alignment: two floor corners of the room outline (indices into twin.room.polygon) that no furniture blocks.
// 0 = south-west corner (the room origin), 5 = north-west corner; they are 7.2 m apart along the west wall.
export const CALIBRATION = [
    { corner: 0, label: 'Corner 1: SOUTH-WEST', hint: 'The floor corner on the conveyor / air-conditioner wall, at the end FAR from the conveyor.' },
    { corner: 5, label: 'Corner 2: NORTH-WEST', hint: 'From corner 1, follow the other wall to its far end: the floor corner on the interactive display (TV) wall.' },
];

// Conveyor mode: two points on the conveyor itself, touched with the controller. Room frame (z-up, metres): the top
// corners of the conveyor frame (CONV-01 in twin.json, top rail at 0.99 m) on its long side facing the room
// (x = 4.647, the east side is against the wall), at the motor (south) end and at the far (north) end, 1.81 m apart.
export const TOUCH_POINTS = [
    { p: [4.647, 0.017, 0.99], label: 'Point 1: MOTOR END', hint: 'The top corner of the conveyor frame at the motor end (by the wall), on the long side that faces the room.' },
    { p: [4.647, 1.823, 0.99], label: 'Point 2: FAR END', hint: 'The top corner at the other end of the same long side.' },
];
// Registration checkpoints (measurement sessions): physical points touched with the yellow ball. The recorded position
// in the room frame is compared with the model position (room frame, z up, metres).
export const CHECKPOINTS = [
    { id: 'CP1', p: [4.647, 0.017, 0.99], label: 'Conveyor frame, MOTOR END', hint: 'Top corner of the frame at the motor end, long side facing the room.' },
    { id: 'CP2', p: [4.647, 1.823, 0.99], label: 'Conveyor frame, FAR END', hint: 'Top corner at the other end of the same long side.' },
    { id: 'CP3', p: [0, 0, 0], label: 'Floor corner SOUTH-WEST', hint: 'Floor corner on the conveyor / air-conditioner wall, far from the conveyor.' },
    { id: 'CP4', p: [0, 7.206, 0], label: 'Floor corner NORTH-WEST', hint: 'Floor corner at the far end of the west wall (TV wall).' },
    { id: 'CP5', p: [5.459, 7.206, 0], label: 'Floor corner NORTH-EAST', hint: 'Floor corner diagonally opposite the south-west corner.' },
];
const ASK_PRESETS = [
    (s) => `Why is the alert level ${s?.machine_status || 'what it is'} right now?`,
    () => 'Summarise the forecast for the next 6 hours in two sentences.',
    () => 'Please draft a work order for the current machine condition.',
];
const TOUCH_OFFSET = 0.05;  // the touch ball sits 5 cm in front of the controller, so the controller body stays clear
const FIT_TOL = 0.25;       // wall fit: a scanned wall within 25 cm of a model wall counts as the same wall
const MODES = ['onsite', 'conveyor', 'tabletop'];
const XRAY_VIEW = { onsite: 1, conveyor: 0 };  // the view shown right after aligning, to check against the real conveyor

// What each view shows. asset: the conveyor and motor ('solid' | 'xray' | 'hidden'); room: the rest of the model.
const PRESETS = {
    onsite: [
        { name: 'Status overlay', asset: 'hidden', room: false, cloud: false },
        { name: 'X-ray conveyor', asset: 'xray', room: false, cloud: false },
        { name: 'Scan check (point cloud)', asset: 'xray', room: false, cloud: true },
        { name: 'Full twin', asset: 'solid', room: true, cloud: false },
    ],
    conveyor: [
        { name: 'X-ray conveyor', asset: 'xray', room: false, cloud: false },
        { name: 'Solid conveyor', asset: 'solid', room: false, cloud: false },
        { name: 'Status overlay', asset: 'hidden', room: false, cloud: false },
    ],
    tabletop: [
        { name: 'Room (cutaway)', asset: 'solid', room: true, cloud: false },
        { name: 'Room + scan point cloud', asset: 'solid', room: true, cloud: true },
        { name: 'Scan point cloud only', asset: 'hidden', room: false, cloud: true },
    ],
};

// ================= CANVAS PANELS =================
function wrap(g, text, max) {
    const out = [];
    let line = '';
    String(text).split(' ').forEach(w => {
        const t = line ? `${line} ${w}` : w;
        if (line && g.measureText(t).width > max) { out.push(line); line = w; } else line = t;
    });
    out.push(line);
    return out;
}

// A text panel whose height follows its content. anchor: which edge of the panel sits at its position.
function makePanel(width, pxW = 1024, pxH = 1024, anchor = 'top') {
    const canvas = document.createElement('canvas');
    canvas.width = pxW;
    canvas.height = pxH;
    const g = canvas.getContext('2d');
    const tex = new THREE.CanvasTexture(canvas);
    tex.colorSpace = THREE.SRGBColorSpace;
    const mesh = new THREE.Mesh(new THREE.PlaneGeometry(1, 1),
        new THREE.MeshBasicMaterial({ map: tex, transparent: true, depthTest: false, depthWrite: false }));
    mesh.renderOrder = 100;
    let used = pxH, w = width;
    const layout = () => {
        const h = (w * used) / pxW;
        mesh.geometry.dispose();
        mesh.geometry = new THREE.PlaneGeometry(w, h).translate(0, anchor === 'top' ? -h / 2 : anchor === 'bottom' ? h / 2 : 0, 0);
        tex.repeat.set(1, used / pxH);
        tex.offset.set(0, 1 - used / pxH);
    };
    // lines: [{ text, size, color, bold, gap, action }] or [{ spark: [values], height, color }]
    // A line with an action is drawn as a button; hit(uv) returns the action of the row under a ray hit.
    let regions = [];
    const draw = (lines, accent = '#5eead4') => {
        const pad = 28;
        const rows = [];
        const sparks = [];
        regions = [];
        let y = pad;
        lines.forEach(l => {
            if (l.spark) {
                const hgt = l.height || 70;
                sparks.push({ v: l.spark, y0: y + 10, y1: y + hgt, color: l.color || '#5eead4' });
                y += hgt + 10 + (l.gap || 0);
                return;
            }
            const size = l.size || 30;
            const y0 = y;
            const font = `${l.bold ? '700' : '400'} ${size}px system-ui, -apple-system, Segoe UI, Roboto, sans-serif`;
            g.font = font;
            wrap(g, l.text, pxW - 2 * pad - (l.action ? 44 : 12)).forEach(t => {
                y += size * 1.28;
                rows.push({ t, y, font, color: l.color || '#e2e8f0', x: l.action ? 20 : 0 });
            });
            if (l.action) { regions.push({ y0: y0 + 2, y1: y + size * 0.42, action: l.action }); y += size * 0.5; }
            y += l.gap || 0;
        });
        used = Math.min(pxH, Math.ceil(y + pad));
        g.clearRect(0, 0, pxW, pxH);
        g.fillStyle = 'rgba(15, 23, 42, 0.88)';
        g.beginPath();
        g.roundRect(2, 2, pxW - 4, used - 4, 24);
        g.fill();
        g.fillStyle = accent;
        g.fillRect(2, 18, 10, used - 36);
        regions.forEach(r => {
            g.fillStyle = 'rgba(94, 234, 212, 0.16)';
            g.beginPath();
            g.roundRect(pad + 4, r.y0, pxW - 2 * pad - 8, r.y1 - r.y0, 14);
            g.fill();
        });
        rows.forEach(r => {
            g.font = r.font;
            g.fillStyle = r.color;
            g.fillText(r.t, pad + 12 + r.x, r.y - 6);
        });
        sparks.forEach(sp => {
            const v = sp.v, lo = Math.min(...v), hi = Math.max(...v), span = hi - lo || 1;
            const x0 = pad + 40, x1 = pxW - pad - 40;
            const pt = (i) => [x0 + (i * (x1 - x0)) / Math.max(1, v.length - 1), sp.y1 - ((v[i] - lo) / span) * (sp.y1 - sp.y0)];
            g.strokeStyle = sp.color;
            g.lineWidth = 5;
            g.beginPath();
            v.forEach((_, i) => { const [x, yy] = pt(i); if (i) g.lineTo(x, yy); else g.moveTo(x, yy); });
            g.stroke();
            g.fillStyle = sp.color;
            v.forEach((_, i) => { const [x, yy] = pt(i); g.beginPath(); g.arc(x, yy, 8, 0, 2 * Math.PI); g.fill(); });
        });
        tex.needsUpdate = true;
        layout();
    };
    const setWidth = (nw) => { w = nw; layout(); };
    const dispose = () => { mesh.geometry.dispose(); mesh.material.dispose(); tex.dispose(); };
    // uv of a ray hit on the plane -> canvas y (the plane shows the top `used` pixels of the canvas)
    const hit = (uv) => {
        const cy = (1 - uv.y) * used;
        return regions.find(r => cy >= r.y0 && cy <= r.y1)?.action || null;
    };
    return { mesh, draw, setWidth, dispose, hit };
}

const shown = (o) => {
    for (; o; o = o.parent) if (!o.visible) return false;
    return true;
};

function readSaved() {
    try { return JSON.parse(localStorage.getItem(STORAGE_KEY) || 'null'); } catch { return null; }
}

// Removes the saved on-site alignment (also callable from the dashboard)
export function clearSavedAlignment() {
    try { localStorage.removeItem(STORAGE_KEY); } catch { /* storage blocked */ }
}

// ================= SESSION =================
// Must be called from a user gesture (button click): the scene is built synchronously, then the session is requested.
export async function startTwinAR({ twin, mode = 'onsite', summary = null, running = true, onEnd }) {
    if (!navigator.xr) throw new Error('WebXR is not available in this browser. Open the dashboard in the Meta Quest Browser.');

    // ---------- renderer and scene ----------
    const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
    renderer.setClearColor(0x000000, 0);  // transparent: the passthrough camera image shows behind the model
    renderer.xr.enabled = true;
    renderer.xr.setReferenceSpaceType('local-floor');
    Object.assign(renderer.domElement.style, { position: 'fixed', width: '1px', height: '1px', opacity: '0', pointerEvents: 'none' });
    document.body.appendChild(renderer.domElement);

    const scene = new THREE.Scene();
    scene.add(new THREE.HemisphereLight('#ffffff', '#8890a0', 2.2));
    const sun = new THREE.DirectionalLight('#ffffff', 1.4);
    sun.position.set(6, 10, 4);
    scene.add(sun);
    const camera = new THREE.PerspectiveCamera(70, 1, 0.02, 100);

    // placement (pose and scale in the real world) > model (y-up room frame, centred for tabletop) > root (z-up room frame)
    const placement = new THREE.Group();
    placement.visible = false;
    scene.add(placement);
    const model = new THREE.Group();
    placement.add(model);
    const root = new THREE.Group();
    root.rotation.x = -Math.PI / 2;
    model.add(root);

    const mats = buildMaterials(twin);
    const groups = {};
    twin.elements.forEach(el => {
        const g = buildElement(el, mats);
        groups[el.id] = g;
        root.add(g);
    });
    const assetIds = new Set([twin.asset, ...twin.sensors.map(s => s.element)]);
    const loops = Object.values(groups).map(g => g.userData.loop).filter(Boolean);
    const walls = wallPlanes(twin);
    root.traverse(o => { if (o.isMesh) o.userData.solid = o.material; });
    const xrayMats = new Map();
    const xrayOf = (m) => {
        if (!xrayMats.has(m)) {
            const x = m.clone();
            Object.assign(x, { transparent: true, opacity: 0.4 * m.opacity, depthWrite: false });
            xrayMats.set(m, x);
        }
        return xrayMats.get(m);
    };

    // Status box around the monitored asset, sensor markers and the live status card
    model.updateMatrixWorld(true);
    const assetBox = new THREE.Box3();
    assetIds.forEach(id => assetBox.expandByObject(groups[id]));
    assetBox.expandByScalar(0.04);
    const haloMat = new THREE.MeshBasicMaterial({ color: NEUTRAL, transparent: true, opacity: 0.15, depthWrite: false });
    const halo = new THREE.Mesh(new THREE.BoxGeometry(...assetBox.getSize(new THREE.Vector3()).toArray()), haloMat);
    halo.position.copy(assetBox.getCenter(new THREE.Vector3()));
    const haloEdges = new THREE.LineSegments(new THREE.EdgesGeometry(halo.geometry), new THREE.LineBasicMaterial({ color: NEUTRAL }));
    halo.add(haloEdges);
    model.add(halo);

    const markers = twin.sensors.map(s => {
        const marker = new THREE.Mesh(new THREE.SphereGeometry(0.03, 16, 12), new THREE.MeshBasicMaterial({ color: NEUTRAL, depthTest: false }));
        marker.renderOrder = 50;
        marker.position.copy(toThree(s.pos));
        marker.userData.sensor = s;
        model.add(marker);
        return marker;
    });
    const cardAnchor = markers.reduce((v, m) => v.add(m.position), new THREE.Vector3()).divideScalar(markers.length);
    cardAnchor.y = assetBox.max.y + 0.6;  // the card hangs above the conveyor, so the machine itself stays visible
    const leaderGeom = new THREE.BufferGeometry().setFromPoints([new THREE.Vector3(), new THREE.Vector3()]);
    const leader = new THREE.Line(leaderGeom, new THREE.LineBasicMaterial({ color: '#e2e8f0', depthTest: false }));
    leader.renderOrder = 99;
    leader.frustumCulled = false;
    scene.add(leader);

    const card = makePanel(0.36, 1024, 640, 'bottom');
    const info = makePanel(0.3, 1024, 768, 'bottom');
    const help = makePanel(0.22, 1024, 1280, 'top');
    const actions = makePanel(0.34, 1024, 1800, 'bottom');
    scene.add(card.mesh, info.mesh, help.mesh, actions.mesh);
    info.mesh.visible = false;
    actions.mesh.visible = false;

    // Aiming reticle and the two alignment pins
    const reticle = new THREE.Mesh(new THREE.RingGeometry(0.035, 0.05, 32).rotateX(-Math.PI / 2),
        new THREE.MeshBasicMaterial({ color: '#ffffff', depthTest: false }));
    reticle.renderOrder = 101;
    reticle.visible = false;
    scene.add(reticle);
    const pins = [0, 1].map(() => {
        const pin = new THREE.Mesh(new THREE.CylinderGeometry(0.008, 0.008, 0.4).translate(0, 0.2, 0),
            new THREE.MeshBasicMaterial({ color: '#fde047', depthTest: false }));
        pin.renderOrder = 101;
        pin.visible = false;
        scene.add(pin);
        return pin;
    });
    // Touch balls (conveyor alignment): one in front of each controller, put on the real corner instead of aiming
    const ballGeom = new THREE.SphereGeometry(0.01, 16, 12);
    const ballMat = new THREE.MeshBasicMaterial({ color: '#fde047', depthTest: false });
    const balls = new Map();
    const ballFor = (src) => {
        if (!balls.has(src)) {
            const ball = new THREE.Mesh(ballGeom, ballMat);
            ball.renderOrder = 102;
            scene.add(ball);
            balls.set(src, ball);
        }
        return balls.get(src);
    };

    // ---------- session (requested while the click's user activation is still valid) ----------
    // plane-detection: the walls and floor of the headset's Space Setup, for the automatic on-site alignment
    const session = await navigator.xr.requestSession('immersive-ar', {
        requiredFeatures: ['local-floor'],
        optionalFeatures: ['hit-test', 'anchors', 'hand-tracking', 'plane-detection'],
    });
    await renderer.xr.setSession(session);
    const refSpace = renderer.xr.getReferenceSpace();

    // Controller rays (three.js controllers follow the target ray of each input source)
    const rayLines = [0, 1].map(i => {
        const c = renderer.xr.getController(i);
        const line = new THREE.Line(new THREE.BufferGeometry().setFromPoints([new THREE.Vector3(), new THREE.Vector3(0, 0, -1)]),
            new THREE.LineBasicMaterial({ color: '#ffffff', transparent: true, opacity: 0.7 }));
        c.add(line);
        c.addEventListener('connected', e => { c.userData.src = e.data; });
        c.addEventListener('disconnected', () => { c.userData.src = null; });
        scene.add(c);
        return { c, line };
    });
    // The help panel rides on the left controller; with hands only it floats below the line of sight
    [0, 1].forEach(i => {
        const grip = renderer.xr.getControllerGrip(i);
        scene.add(grip);
        grip.addEventListener('connected', e => {
            if (e.data.handedness !== 'left' || e.data.hand) return;
            grip.add(help.mesh);
            help.mesh.position.set(0, 0.06, -0.02);
            help.mesh.rotation.set(-Math.PI / 3, 0, 0);
        });
        grip.addEventListener('disconnected', () => { if (help.mesh.parent === grip) scene.add(help.mesh); });
    });

    // ---------- state ----------
    const poly = twin.room.polygon;
    const roomCenter = [0, 1].map(k => (Math.min(...poly.map(p => p[k])) + Math.max(...poly.map(p => p[k]))) / 2);
    // align: how the current 1:1 pose was made ('auto' | 'corners' | 'touch' | 'restored'); fits: the wall fits of
    // the last automatic alignment, best first (X steps through them); alignInfo: the quality line on the help panel
    const S = {
        mode, preset: 0, step: 'place', calPoints: [], scale: 1 / 12,
        align: null, alignInfo: null, fits: null, fitIndex: 0, autoStart: 0,
        summary, running, paused: false, ended: false,
        onsitePose: null, anchor: null, anchorUuid: null, anchorStatus: '', wantAnchor: false, reanchorAt: 0,
        cloud: null, cloudState: 'idle', message: '', activeSource: null, lastTime: 0, helpKey: '',
        // human in the loop: actions menu, the action in progress, its result line and the last copilot answer
        menu: false, busy: null, actionStatus: '', answer: '', chatSession: `mr-${Math.random().toString(36).slice(2, 10)}`,
        // measurement: checkpoint mode, next checkpoint, last result; alignment start; frame-time samples
        measure: false, cpIndex: 0, cpLast: null, alignStartedAt: 0, frames: [], fpsSince: 0, startedAt: performance.now(),
    };
    const hitSources = new Map(), targets = new Map(), rays = new Map(), prevButtons = new Map();
    const head = { pos: new THREE.Vector3(0, 1.6, 0), fwd: new THREE.Vector3(0, 0, -1), yaw: 0 };
    const oneToOne = () => S.mode !== 'tabletop';

    const preset = () => PRESETS[S.mode][S.preset];
    const flash = (msg, ms = 4000) => {
        S.message = msg;
        drawHelp();
        setTimeout(() => { if (S.message === msg) { S.message = ''; drawHelp(); } }, ms);
    };

    const savePose = () => {
        try {
            localStorage.setItem(STORAGE_KEY, JSON.stringify({
                pos: placement.position.toArray(), quat: placement.quaternion.toArray(),
                uuid: S.anchorUuid, method: S.align, info: S.alignInfo, savedAt: new Date().toISOString(),
            }));
        } catch { /* storage blocked: the alignment then lasts for this session only */ }
    };

    function setScaleDependent() {
        const s = placement.scale.x;
        markers.forEach(m => m.scale.setScalar((oneToOne() ? 0.025 : 0.006) / (0.03 * s)));
        if (S.cloud) S.cloud[0].material.size = CLOUD_POINT_SIZE * s;
        card.setWidth(oneToOne() ? 0.36 : 0.16);
        actions.setWidth(oneToOne() ? 0.34 : 0.18);
        info.setWidth(oneToOne() ? 0.3 : 0.18);
    }

    function applyPreset() {
        const P = preset();
        root.traverse(o => {
            if (!o.isMesh || !assetIds.has(o.userData.elementId)) return;
            o.material = P.asset === 'xray' ? xrayOf(o.userData.solid) : o.userData.solid;
        });
        if (P.cloud && S.cloudState === 'idle') {
            S.cloudState = 'loading';
            loadCloud(CLOUD_POINT_SIZE).then(segs => {
                if (S.ended) return;
                S.cloud = segs;
                segs.forEach(seg => root.add(seg));
                S.cloudState = 'ready';
                setScaleDependent();
                drawHelp();
            }).catch(() => { S.cloudState = 'error'; drawHelp(); });
        }
        drawHelp();
    }

    // ---------- 1:1 alignment ----------
    // steps: 'auto' (wall fit, on-site), 'calibrate' (two floor corners with the ray), 'touch' (two conveyor corners)
    function beginAlign(step) {
        S.step = step;
        S.alignStartedAt = performance.now();
        S.calPoints = [];
        S.anchor = null;
        S.autoStart = 0;
        placement.visible = false;
        pins.forEach(p => { p.visible = false; });
        info.mesh.visible = false;
        drawHelp();
    }
    const startCalibration = () => beginAlign('calibrate');
    const startTouch = () => beginAlign('touch');
    const startAutoAlign = () => { S.fits = null; beginAlign('auto'); };

    // X button: on-site steps from the automatic fit to the next-best fit, then to the two corners, then back to auto
    function realign() {
        if (S.mode === 'tabletop') { S.step = 'place'; drawHelp(); return; }
        if (S.mode === 'conveyor') { startTouch(); return; }
        if (S.step === 'calibrate') { startCalibration(); return; }
        if (S.step === 'auto') return;
        if (S.align === 'auto' && S.fits && S.fitIndex + 1 < S.fits.length) applyFit(S.fitIndex + 1);
        else if (S.align === 'auto') startCalibration();
        else startAutoAlign();
    }
    function realignLabel() {
        if (S.mode === 'tabletop') return 're-place';
        if (S.mode === 'conveyor') return 'touch-align again';
        if (S.step === 'calibrate') return 'restart corners';
        if (S.align === 'auto') return S.fits && S.fitIndex + 1 < S.fits.length ? 'next fit' : 'align with 2 corners';
        return 'auto-align';
    }

    function applyMode(m) {
        S.mode = m;
        S.preset = 0;
        info.mesh.visible = false;
        pins.forEach(p => { p.visible = false; });
        if (m === 'tabletop') {
            model.position.copy(toThree([-roomCenter[0], -roomCenter[1], 0]));
            placement.scale.setScalar(S.scale);
            placement.visible = false;
            S.step = 'place';
        } else {
            model.position.set(0, 0, 0);
            placement.scale.setScalar(1);
            if (S.onsitePose) {
                placement.position.copy(S.onsitePose.pos);
                placement.quaternion.copy(S.onsitePose.quat);
                placement.visible = true;
                S.step = 'placed';
            } else if (m === 'conveyor') startTouch();
            else startAutoAlign();
        }
        setScaleDependent();
        applyPreset();
    }

    function finishAlign(pos, quat, method, alignInfo, message) {
        placement.position.copy(pos);
        placement.quaternion.copy(quat);
        S.onsitePose = { pos: pos.clone(), quat: quat.clone() };
        S.anchor = null;  // the new pose gets a new anchor (served in the frame loop)
        S.align = method;
        S.alignInfo = alignInfo;
        placement.visible = true;
        pins.forEach(p => { p.visible = false; });
        S.step = 'placed';
        S.preset = XRAY_VIEW[S.mode];  // x-ray first, so the alignment can be checked against the real conveyor
        applyPreset();
        S.wantAnchor = true;
        logRecord('align', {
            method, mode: S.mode, info: alignInfo?.text || null, fits: S.fits?.length ?? null, fit_index: S.fitIndex,
            align_ms: S.alignStartedAt ? Math.round(performance.now() - S.alignStartedAt) : null,
            pose: { pos: pos.toArray(), quat: quat.toArray() },
        });
        flash(message, 7000);
    }

    // Two measured points p1, p2 (world) <-> the same points q1, q2 in the model: yaw + translation, no scale.
    // The difference between the measured and the modelled distance is reported as a quality check.
    function solveTwoPoints(method) {
        const touch = method === 'touch';
        const [p1, p2] = S.calPoints;
        const [q1, q2] = touch ? TOUCH_POINTS.map(t => toThree(t.p)) : CALIBRATION.map(c => toThree([...poly[c.corner], 0]));
        const theta = Math.atan2(q2.z - q1.z, q2.x - q1.x) - Math.atan2(p2.z - p1.z, p2.x - p1.x);
        const quat = new THREE.Quaternion().setFromAxisAngle(Y_AXIS, theta);
        const r1 = q1.clone().applyQuaternion(quat), r2 = q2.clone().applyQuaternion(quat);
        const pos = new THREE.Vector3((p1.x - r1.x + p2.x - r2.x) / 2, (p1.y - r1.y + p2.y - r2.y) / 2, (p1.z - r1.z + p2.z - r2.z) / 2);
        const cm = Math.abs(Math.hypot(p2.x - p1.x, p2.z - p1.z) - Math.hypot(q2.x - q1.x, q2.z - q1.z)) * 100;
        const [good, fair] = touch ? [1.5, 3] : [5, 10];
        const alignInfo = { text: `${touch ? 'Point' : 'Corner'} distance error: ${cm.toFixed(1)} cm`, color: cm <= good ? '#86efac' : cm <= fair ? '#fde047' : '#fca5a5' };
        finishAlign(pos, quat, method, alignInfo, cm > fair
            ? `The distance is off by ${cm.toFixed(0)} cm: check the ${touch ? 'points' : 'corners'} and press X to align again.`
            : touch ? 'Aligned on the conveyor. Fine-tune with the thumbsticks if needed.'
                : 'Aligned. The x-ray conveyor should sit on the real one; fine-tune with the thumbsticks.');
    }

    // ---------- automatic alignment to the Space Setup walls ----------
    // Model walls (room outline) in the three.js floor plane (x, z) with their inward normals
    const modelWalls = walls.map((w, i) => {
        const b = poly[(i + 1) % poly.length];
        return { a: [w.a[0], -w.a[1]], b: [b[0], -b[1]], n: [w.n[0], -w.n[1]] };
    });
    const dot2 = (a, b) => a[0] * b[0] + a[1] * b[1];

    // Fits the room outline to the scanned walls: the yaw from the dominant wall direction (mod 90 deg), then, for each
    // of the four 90-deg turns, the translation along both room axes that puts the most scanned wall length on model
    // walls. Returns the fits whose matched wall length is at least 60% of the best one, best first (null if none).
    function fitToWalls(frame) {
        const det = [];
        let floorY = null;
        const m4 = new THREE.Matrix4();
        for (const plane of frame.detectedPlanes) {
            const pose = frame.getPose(plane.planeSpace, refSpace);
            if (!pose) continue;
            const label = (plane.semanticLabel || '').toLowerCase();
            if (plane.orientation === 'horizontal') {
                if (label === 'floor') floorY = pose.transform.position.y;
                continue;
            }
            if (label && label !== 'wall') continue;  // door and window frames, furniture
            m4.fromArray(pose.transform.matrix);
            const n = new THREE.Vector3(0, 1, 0).applyQuaternion(tmpQ.copy(pose.transform.orientation)).setY(0);
            if (n.lengthSq() < 0.8) continue;  // not vertical
            n.normalize();
            const pts = plane.polygon.map(p => new THREE.Vector3(p.x, p.y, p.z).applyMatrix4(m4));
            if (pts.length < 2) continue;
            const c = pts.reduce((sum, p) => sum.add(p), new THREE.Vector3()).divideScalar(pts.length);
            if ((head.pos.x - c.x) * n.x + (head.pos.z - c.z) * n.z < 0) n.negate();  // inward: towards the user in the room
            const d = [-n.z, n.x];
            const s = pts.map(p => p.x * d[0] + p.z * d[1]);
            const s0 = Math.min(...s), s1 = Math.max(...s);
            if (s1 - s0 < 0.4) continue;
            det.push({ n: [n.x, n.z], c: c.x * n.x + c.z * n.z, s0, s1, len: s1 - s0 });
        }
        if (det.length < 2) return null;
        const total = det.reduce((sum, w) => sum + w.len, 0);
        let sx = 0, sy = 0;
        det.forEach(w => { const a = 4 * Math.atan2(w.n[1], w.n[0]); sx += w.len * Math.cos(a); sy += w.len * Math.sin(a); });
        const phi = Math.atan2(sy, sx) / 4;  // the model's axis-aligned walls turned by theta point at -theta (mod 90 deg)
        // Walls facing fewer than three directions (e.g. one corner of the room) do not fix the position: no fit
        const facing = new Set(det.map(w => ((Math.round((Math.atan2(w.n[1], w.n[0]) - phi) / (Math.PI / 2)) % 4) + 4) % 4));
        if (facing.size < 3) return null;

        const fits = [];
        for (let k = 0; k < 4; k++) {
            const quat = new THREE.Quaternion().setFromAxisAngle(Y_AXIS, -phi + (k * Math.PI) / 2);
            const rot = ([x, z]) => { const v = new THREE.Vector3(x, 0, z).applyQuaternion(quat); return [v.x, v.z]; };
            const mw = modelWalls.map(w => {
                const a = rot(w.a), b = rot(w.b), n = rot(w.n), d = [-n[1], n[0]];
                const sa = dot2(a, d), sb = dot2(b, d);
                return { n, d, c: dot2(a, n), s0: Math.min(sa, sb), s1: Math.max(sa, sb) };
            });
            const pairs = [];  // (scanned wall, model wall) with the same inward normal
            det.forEach(w => mw.forEach(m => { if (dot2(w.n, m.n) > 0.97) pairs.push({ w, m }); }));
            const axes = [rot([1, 0]), rot([0, 1])];
            const t = [];
            for (const u of axes) {
                const along = pairs.filter(p => Math.abs(dot2(p.m.n, u)) > 0.9)
                    .map(p => ({ w: p.w, tu: (p.w.c - p.m.c) * Math.sign(dot2(p.m.n, u)) }));
                if (!along.length) break;
                // each scanned wall counts once, with its pair closest to the candidate shift
                const inliers = (tu) => {
                    const byWall = new Map();
                    along.forEach(p => {
                        const r = Math.abs(p.tu - tu);
                        if (r < FIT_TOL && (!byWall.has(p.w) || r < Math.abs(byWall.get(p.w).tu - tu))) byWall.set(p.w, p);
                    });
                    return [...byWall.values()];
                };
                const weight = (ps) => ps.reduce((sum, p) => sum + p.w.len, 0);
                let best = along[0].tu, bestW = -1;
                along.forEach(p => { const wt = weight(inliers(p.tu)); if (wt > bestW) { bestW = wt; best = p.tu; } });
                const inl = inliers(best);
                t.push(inl.reduce((sum, p) => sum + p.tu * p.w.len, 0) / weight(inl));
            }
            if (t.length < 2) continue;  // scanned walls in one direction only: the shift along the other is unknown
            const T = [axes[0][0] * t[0] + axes[1][0] * t[1], axes[0][1] * t[0] + axes[1][1] * t[1]];
            let matched = 0;
            det.forEach(w => {
                let ov = 0;
                mw.forEach(m => {
                    if (dot2(w.n, m.n) < 0.97 || Math.abs(w.c - m.c - dot2(T, m.n)) > FIT_TOL) return;
                    const sh = dot2(T, m.d);
                    ov = Math.max(ov, Math.min(w.s1, m.s1 + sh) - Math.max(w.s0, m.s0 + sh));
                });
                matched += ov;
            });
            fits.push({ quat, pos: new THREE.Vector3(T[0], floorY ?? 0, T[1]), matched, coverage: matched / total, walls: det.length });
        }
        fits.sort((a, b) => b.matched - a.matched);
        if (!fits.length || fits[0].matched < Math.max(1.5, 0.3 * total)) return null;
        return fits.filter(f => f.matched >= 0.6 * fits[0].matched);
    }

    function applyFit(i) {
        const f = S.fits[i];
        S.fitIndex = i;
        const more = i + 1 < S.fits.length;
        const pct = Math.round(f.coverage * 100);
        finishAlign(f.pos, f.quat, 'auto',
            { text: `Auto-aligned to ${f.walls} scanned walls, ${pct}% matched${S.fits.length > 1 ? ` (fit ${i + 1} of ${S.fits.length})` : ''}`, color: pct >= 60 ? '#86efac' : '#fde047' },
            `Aligned to the room scan. If the twin is off or turned round, press X for ${more ? 'the next fit' : 'the two-corner alignment'}.`);
    }

    // Runs in the frame loop while S.step === 'auto'. The planes arrive over the first frames, so it waits a second;
    // without plane detection (or without usable walls after 5 s) it falls back to the two corners.
    function autoAlign(time, frame) {
        if (!S.autoStart) S.autoStart = time;
        const waited = time - S.autoStart;
        if (waited < 1000) return;
        if (!frame.detectedPlanes) {
            startCalibration();
            flash('No room data (allow it in the browser, or run Space Setup): align with two corners.', 7000);
            return;
        }
        let fits = frame.detectedPlanes.size ? fitToWalls(frame) : null;
        if (fits) {
            // Near-equal fits (a scan that misses the room's small notch matches it turned round as well): the one
            // closest to the previous alignment goes first; X still steps to the others
            const prev = S.onsitePose?.quat;
            if (prev && fits.length > 1) {
                const near = fits.filter(f => f.matched >= 0.9 * fits[0].matched).sort((a, b) => a.quat.angleTo(prev) - b.quat.angleTo(prev));
                fits = [...near, ...fits.filter(f => !near.includes(f))];
            }
            S.fits = fits;
            applyFit(0);
        }
        else if (waited > 5000) {
            startCalibration();
            flash('The scanned walls do not match the room: align with two corners (or redo Space Setup).', 7000);
        }
    }

    function onSelect(src) {
        S.activeSource = src;
        if (S.step === 'placed') {
            const a = S.menu && hitPanel(actions, src);
            if (a) { runAction(a); return; }
            if (S.measure) { recordCheckpoint(src); return; }
        }
        const point = targets.get(src);
        if (S.step === 'calibrate' || S.step === 'touch') {
            const touch = S.step === 'touch';
            const ray = rays.get(src);
            const p = touch ? ray && ray.origin.clone().addScaledVector(ray.dir, TOUCH_OFFSET) : point;
            if (!p) return flash(touch ? 'Controller not tracked: hold it where the headset can see it.' : 'No surface found: aim the ray at the floor corner.');
            pins[S.calPoints.length].position.copy(p);
            pins[S.calPoints.length].visible = true;
            S.calPoints.push(p.clone());
            if (S.calPoints.length === 2) solveTwoPoints(touch ? 'touch' : 'corners'); else drawHelp();
        } else if (S.step === 'auto') {
            flash('Looking for the scanned walls...');
        } else if (S.step === 'place') {
            if (!point) return flash('No surface found: aim at a table or the floor.');
            S.step = 'placed';
            drawHelp();
        } else pick(src);
    }
    session.addEventListener('select', e => onSelect(e.inputSource));

    // Point at an element (or a sensor) to show its details
    const raycaster = new THREE.Raycaster();
    function pick(src) {
        const ray = rays.get(src);
        if (!ray) return;
        raycaster.set(ray.origin, ray.dir);
        const hit = raycaster.intersectObjects([...markers, ...Object.values(groups)], true).find(h => shown(h.object));
        if (!hit) { info.mesh.visible = false; S.selected = null; return; }
        const sensor = hit.object.userData.sensor;
        const id = sensor ? sensor.element : hit.object.userData.elementId;
        S.selected = { el: groups[id].userData.element, sensor };
        info.mesh.position.copy(hit.point);
        info.mesh.visible = true;
        drawInfo();
    }

    function drawInfo() {
        if (!S.selected) return;
        const { el, sensor } = S.selected;
        const lines = [
            { text: sensor ? sensor.label : el.name, bold: true, size: 36, color: '#f8fafc' },
            { text: `${el.category} · Mark ${el.mark}`, size: 26, color: '#94a3b8', gap: 8 },
        ];
        if (el.note) lines.push({ text: el.note, size: 26, color: '#cbd5e1', gap: 8 });
        const metrics = sensor ? sensor.metrics : assetIds.has(el.id) ? twin.sensors.flatMap(s => s.metrics) : [];
        metrics.forEach(m => lines.push({ text: `${m.label}: ${fmt(S.summary?.metrics?.[m.key])} ${m.unit}`, size: 30, color: '#f1f5f9' }));
        info.draw(lines, assetIds.has(el.id) ? ZONE_COLORS[S.summary?.iso_zone] || NEUTRAL : '#94a3b8');
    }

    function drawCard() {
        const s = S.summary;
        const zone = ZONE_COLORS[s?.iso_zone] || NEUTRAL;
        const flagged = s?.shift_episode?.currently_flagged || [];
        haloMat.color.set(zone);
        haloEdges.material.color.set(zone);
        markers.forEach(m => m.material.color.set(sensorColor(m.userData.sensor, s)));
        const lines = [
            { text: `${twin.asset} · Zone ${s?.iso_zone ?? '–'}${s?.status ? ` - ${s.status}` : ''}`, bold: true, size: 40, color: zone },
            { text: `Alert level: ${s?.machine_status ?? '–'}`, size: 28, color: '#cbd5e1', gap: (s?.alert_reasons || []).length ? 0 : 10 },
            ...((s?.alert_reasons || []).slice(0, 1).map(r => ({ text: r, size: 22, color: '#94a3b8', gap: 10 }))),
            ...twin.sensors.flatMap(x => x.metrics).map(m => {
                const warn = flagged.includes(m.key);
                return { text: `${m.label}: ${fmt(s?.metrics?.[m.key])} ${m.unit}${warn ? '  (shift flagged)' : ''}`, size: 32, color: warn ? FLAGGED : '#f1f5f9' };
            }),
            { text: `Reading: ${s?.reading_time || s?.data_timestamp || 'waiting for data'}`, size: 22, color: '#94a3b8' },
        ];
        if (!S.running) lines.push({ text: 'No live data: the sensor feed is stale', size: 22, color: FLAGGED });
        card.draw(lines, zone);
        drawInfo();
        if (S.menu) drawActions();
    }

    // ---------- actions menu (human in the loop) ----------
    function drawActions() {
        const s = S.summary;
        const L = [{ text: 'Maintenance actions', bold: true, size: 38, color: '#5eead4' }];
        L.push({ text: `Alert level: ${s?.machine_status ?? '–'}`, bold: true, size: 30, color: ZONE_COLORS[s?.iso_zone] || '#e2e8f0' });
        (s?.alert_reasons || []).slice(0, 3).forEach(r => L.push({ text: `· ${r}`, size: 23, color: '#cbd5e1' }));
        const f = s?.forecast?.z_rms;
        if (f) {
            L.push({ text: `Z RMS forecast (mm/s): now ${fmt(f.now)} · +1 h ${fmt(f['+1 h'])} · +3 h ${fmt(f['+3 h'])} · +6 h ${fmt(f['+6 h'])}`, size: 25, color: '#f1f5f9', gap: 4 });
            L.push({ spark: [f.now, f['+1 h'], f['+3 h'], f['+6 h']], height: 70, color: ZONE_COLORS[s?.iso_zone] || NEUTRAL, gap: 4 });
        }
        const flagged = Object.entries(s?.shift_scores || {}).filter(([, v]) => v.flagged);
        L.push({ text: flagged.length ? `Shift flagged (s-IDK²): ${flagged.map(([k, v]) => `${k} ${v.score.toFixed(2)} < ${v.threshold.toFixed(2)}`).join(', ')}` : 'No channel flagged by the shift detector', size: 24, color: flagged.length ? FLAGGED : '#94a3b8' });
        const ep = s?.shift_episode;
        if (ep) L.push({ text: `Shift episode since ${ep.since}: ${ep.reviewed ? `reviewed by ${ep.reviewed_by}` : 'NOT reviewed'}`, size: 24, color: ep.reviewed ? '#94a3b8' : FLAGGED });
        L.push({ text: '', size: 10, gap: 6 });
        const noToken = !approverToken();
        const needs = noToken ? '  (enter the approver token on the dashboard first)' : '';
        if (ep && !ep.reviewed) L.push({ text: `Acknowledge the shift episode${needs}`, size: 30, bold: true, action: { kind: 'ack' }, gap: 8 });
        (s?.drafts || []).slice(0, 2).forEach(d => {
            L.push({ text: `Draft: ${d.title || d.session_id}`, size: 24, color: '#cbd5e1' });
            L.push({ text: `Approve this draft${needs}`, size: 28, bold: true, action: { kind: 'approve', session: d.session_id }, gap: 4 });
            L.push({ text: `Reject this draft${needs}`, size: 28, bold: true, action: { kind: 'reject', session: d.session_id }, gap: 8 });
        });
        ASK_PRESETS.forEach(q => { const text = q(s); L.push({ text: `Ask: ${text}`, size: 27, action: { kind: 'ask', text }, gap: 6 }); });
        if (S.actionStatus) L.push({ text: S.actionStatus, size: 24, color: S.busy ? '#fde047' : '#86efac', gap: 6 });
        if (S.answer) L.push({ text: S.answer.replace(/[*#]/g, '').slice(0, 700), size: 24, color: '#e2e8f0', gap: 6 });
        L.push({ text: 'TRIGGER on a row: run it   ·   right GRIP: close', size: 22, color: '#94a3b8' });
        actions.draw(L, '#5eead4');
    }

    const sleep = (ms) => new Promise(r => setTimeout(r, ms));
    function applySummary(next) { S.summary = next; if (!S.ended) drawCard(); }

    // Runs one action; the round trip ends when /api/summary shows the action (its hitl sequence number), which is
    // the state every client, including this one, displays.
    async function runAction(a) {
        if (S.busy) return flash('Waiting for the previous action to finish...');
        if (a.kind !== 'ask' && !approverToken()) return flash('Enter the approver token on the dashboard first (it asks when you review or approve there).', 6000);
        S.busy = a.kind;
        S.actionStatus = a.kind === 'ask' ? 'Asking the copilot...' : 'Sending...';
        drawActions();
        let r;
        if (a.kind === 'ack') r = await timedPost('/api/shift/review', { note: 'Reviewed in mixed reality at the conveyor', source: 'mixed_reality' });
        else if (a.kind === 'approve') r = await timedPost('/api/work_orders/approve', { session_id: a.session, source: 'mixed_reality' });
        else if (a.kind === 'reject') r = await timedPost('/api/work_orders/reject', { session_id: a.session, reason: 'Rejected in mixed reality at the conveyor', source: 'mixed_reality' });
        else r = await timedPost('/api/chat', { message: a.text, session_id: S.chatSession }, { token: false });
        let confirmMs = null;
        if (r.ok && a.kind !== 'ask') {
            const want = r.json?.hitl_seq ?? Infinity;
            for (let i = 0; i < 40 && !S.ended; i++) {
                try {
                    const next = await fetchSummary();
                    if ((next.hitl?.seq ?? 0) >= want) { confirmMs = performance.now() - r.t_cmd; applySummary(next); break; }
                } catch { /* retry */ }
                await sleep(250);
            }
        } else if (r.ok) {
            S.answer = r.json?.response || '';
            confirmMs = r.rtt_ms;
            try { applySummary(await fetchSummary()); } catch { /* drafts refresh on the next poll */ }
        }
        const label = { ack: 'Shift episode acknowledged', approve: 'Draft approved', reject: 'Draft rejected', ask: 'Copilot answered' }[a.kind];
        S.actionStatus = r.ok
            ? `${label}: response ${Math.round(r.rtt_ms)} ms${confirmMs != null && a.kind !== 'ask' ? `, shown in the shared state after ${Math.round(confirmMs)} ms` : ''}`
            : `Failed (${r.status || 'network'}): ${r.json?.detail || r.error || 'no response'}`;
        logRecord('action', {
            action: a.kind, ok: r.ok, status: r.status, rtt_ms: r.rtt_ms, server_ms: r.server_ms, confirm_ms: confirmMs,
            error: r.ok ? null : (r.json?.detail || r.error), mode: S.mode,
        });
        S.busy = null;
        drawActions();
    }

    function hitPanel(panel, src) {
        const ray = rays.get(src);
        if (!ray || !panel.mesh.visible) return null;
        raycaster.set(ray.origin, ray.dir);
        const h = raycaster.intersectObject(panel.mesh)[0];
        return h?.uv ? panel.hit(h.uv) : null;
    }

    // ---------- registration checkpoints (measurement sessions) ----------
    function recordCheckpoint(src) {
        const ray = rays.get(src);
        if (!ray) return flash('Controller not tracked: hold it where the headset can see it.');
        const p = ray.origin.clone().addScaledVector(ray.dir, TOUCH_OFFSET);
        root.updateMatrixWorld(true);
        const q = root.worldToLocal(p.clone());  // room frame, z up, metres
        const cp = CHECKPOINTS[S.cpIndex];
        const e = [q.x - cp.p[0], q.y - cp.p[1], q.z - cp.p[2]];
        const h = Math.hypot(e[0], e[1]), d3 = Math.hypot(...e);
        S.cpLast = `${cp.id}: horizontal ${(h * 100).toFixed(1)} cm, vertical ${(e[2] * 100).toFixed(1)} cm`;
        logRecord('checkpoint', {
            cp: cp.id, expected: cp.p, measured: [q.x, q.y, q.z], err_xyz_m: e, err_h_cm: h * 100, err_3d_cm: d3 * 100,
            mode: S.mode, align: S.align, align_info: S.alignInfo?.text || null, anchor: S.anchorStatus,
            session_s: (performance.now() - S.startedAt) / 1000, head_dist_m: head.pos.distanceTo(p),
        });
        S.cpIndex = (S.cpIndex + 1) % CHECKPOINTS.length;
        drawHelp();
    }

    function drawHelp() {
        const L = [];
        const P = preset();
        const title = { onsite: 'ON-SITE  ·  1:1', conveyor: 'CONVEYOR  ·  1:1', tabletop: `TABLETOP  ·  1:${Math.round(1 / S.scale)}` };
        L.push({ text: title[S.mode], bold: true, size: 40, color: '#5eead4', gap: 6 });
        if (S.step === 'calibrate') {
            const c = CALIBRATION[S.calPoints.length];
            L.push({ text: `Align, step ${S.calPoints.length + 1} of 2`, bold: true, size: 32 });
            L.push({ text: c.label, bold: true, size: 32, color: '#fde047' });
            L.push({ text: c.hint, size: 28, color: '#cbd5e1' });
            L.push({ text: 'Aim the ray exactly at that corner on the floor and pull the TRIGGER.', size: 28, gap: 10 });
        } else if (S.step === 'touch') {
            const t = TOUCH_POINTS[S.calPoints.length];
            L.push({ text: `Align on the conveyor, step ${S.calPoints.length + 1} of 2`, bold: true, size: 32 });
            L.push({ text: t.label, bold: true, size: 32, color: '#fde047' });
            L.push({ text: t.hint, size: 28, color: '#cbd5e1' });
            L.push({ text: 'Hold the controller so the YELLOW BALL sits exactly on that corner, then pull the TRIGGER.', size: 28, gap: 10 });
        } else if (S.step === 'auto') {
            L.push({ text: 'Aligning to the room scan...', bold: true, size: 32 });
            L.push({ text: 'Look around the room for a moment. If the browser asks for room data, choose Allow.', size: 28, gap: 10 });
        } else if (S.step === 'place') {
            L.push({ text: 'Aim at a table or the floor. Pull the TRIGGER to put the model down.', size: 30, gap: 10 });
        } else {
            L.push({ text: `View ${S.preset + 1}/${PRESETS[S.mode].length}: ${P.name}`, bold: true, size: 32 });
            if (P.cloud && S.cloudState !== 'ready') L.push({ text: S.cloudState === 'error' ? 'Point cloud failed to load' : 'Loading point cloud (5 MB)...', size: 26, color: '#fde047' });
            if (oneToOne()) {
                if (S.alignInfo) L.push({ text: S.alignInfo.text, size: 26, color: S.alignInfo.color });
                if (S.anchorStatus) L.push({ text: S.anchorStatus, size: 26, color: '#94a3b8' });
            }
            if (S.measure) {
                const cp = CHECKPOINTS[S.cpIndex];
                L.push({ text: `CHECKPOINT ${S.cpIndex + 1}/${CHECKPOINTS.length}: ${cp.label}`, bold: true, size: 30, color: '#fde047' });
                L.push({ text: `${cp.hint} Put the YELLOW BALL on it and pull the TRIGGER. Right stick click: skip.`, size: 26, color: '#cbd5e1' });
                if (S.cpLast) L.push({ text: `Last: ${S.cpLast}`, size: 26, color: '#86efac', gap: 10 });
            } else L.push({ text: 'TRIGGER on an object or sensor: details', size: 28, gap: 10 });
        }
        if (S.message) L.push({ text: S.message, size: 28, color: '#fca5a5', gap: 10 });
        L.push({ text: `RIGHT   A: next view   B: next mode   GRIP: ${S.menu ? 'close' : 'open'} actions`, size: 26, color: '#e2e8f0' });
        L.push({ text: `             stick: rotate  ·  ${oneToOne() ? 'up / down' : 'bigger / smaller'}`, size: 26, color: '#e2e8f0' });
        L.push({ text: `LEFT     X: ${realignLabel()}   Y: ${S.paused ? 'run' : 'pause'} animation`, size: 26, color: '#e2e8f0' });
        L.push({ text: '             stick: move   ·   click stick: exit AR', size: 26, color: '#e2e8f0' });
        if (MEASURE) L.push({ text: `LEFT GRIP: checkpoints ${S.measure ? 'off' : 'on'} (measurement session)`, size: 24, color: '#fde047' });
        help.draw(L);
    }

    // Anchor handling: frame.createAnchor needs an XRFrame, so requests are queued here and served in the frame loop
    function serveAnchor(frame) {
        if (!S.wantAnchor) return;
        S.wantAnchor = false;
        S.onsitePose = { pos: placement.position.clone(), quat: placement.quaternion.clone() };
        if (!frame.createAnchor) {
            S.anchorStatus = 'Saved on this headset (no anchor support)';
            savePose();
            drawHelp();
            return;
        }
        const p = placement.position, q = placement.quaternion;
        frame.createAnchor(new XRRigidTransform({ x: p.x, y: p.y, z: p.z }, { x: q.x, y: q.y, z: q.z, w: q.w }), refSpace).then(async anchor => {
            if (S.ended) return;
            const oldUuid = S.anchorUuid;
            S.anchor = anchor;
            S.anchorStatus = 'Anchored';
            if (anchor.requestPersistentHandle) {
                try {
                    S.anchorUuid = await anchor.requestPersistentHandle();
                    if (oldUuid && oldUuid !== S.anchorUuid) session.deletePersistentAnchor?.(oldUuid).catch(() => {});
                    S.anchorStatus = 'Alignment saved (persistent anchor)';
                } catch {
                    S.anchorStatus = 'Anchored for this session only';
                }
            }
            savePose();
            drawHelp();
        }).catch(() => {
            S.anchorStatus = 'Anchor failed; alignment saved as a pose';
            savePose();
            drawHelp();
        });
    }

    // Hit-test source per input source (controller ray or hand ray). Needs the room scanned in Space Setup;
    // without it, the ray is intersected with the floor (y = 0 of the local-floor space) instead.
    async function refreshHitSources() {
        const live = [...session.inputSources];
        for (const [src, hs] of hitSources) if (!live.includes(src)) { hs?.cancel(); hitSources.delete(src); }
        if (!session.requestHitTestSource) return;
        for (const src of live) {
            if (hitSources.has(src)) continue;
            hitSources.set(src, null);
            try { hitSources.set(src, await session.requestHitTestSource({ space: src.targetRaySpace })); } catch { /* floor fallback */ }
        }
    }
    session.addEventListener('inputsourceschange', refreshHitSources);
    refreshHitSources();

    // Thumbstick fine-tuning. On-site: rotation pivots on the conveyor, so it does not swing away while aligning.
    const tmpQ = new THREE.Quaternion();
    function rotateAbout(pivot, angle) {
        tmpQ.setFromAxisAngle(Y_AXIS, angle);
        placement.position.sub(pivot).applyQuaternion(tmpQ).add(pivot);
        placement.quaternion.premultiply(tmpQ);
    }
    function adjusted(time) {
        if (!oneToOne()) return;
        S.anchor = null;  // stop following the old anchor; a new one is made when the sticks are released
        S.reanchorAt = time + 1000;
        S.anchorStatus = 'Adjusting...';
    }

    function handleGamepads(time, dt) {
        const dead = (v) => (Math.abs(v) < 0.15 ? 0 : v);
        for (const src of session.inputSources) {
            const gp = src.gamepad;
            if (!gp || src.hand) continue;
            const pressed = gp.buttons.map(b => b.pressed);
            const prev = prevButtons.get(src) || [];
            const edge = (i) => pressed[i] && !prev[i];
            prevButtons.set(src, pressed);
            const ax = dead(gp.axes[2] || 0), ay = dead(gp.axes[3] || 0);
            const placed = S.step === 'placed';
            if (src.handedness === 'right') {
                if (edge(1) && placed) { S.menu = !S.menu; actions.mesh.visible = S.menu; if (S.menu) drawActions(); drawHelp(); }
                if (edge(3) && placed && S.measure) { S.cpIndex = (S.cpIndex + 1) % CHECKPOINTS.length; drawHelp(); }
                else if (edge(3) && placed && MEASURE && S.mode === 'onsite') startAutoAlign();  // repeat trials of the wall fit
                if (edge(4) && placed) { S.preset = (S.preset + 1) % PRESETS[S.mode].length; applyPreset(); }
                if (edge(5)) applyMode(MODES[(MODES.indexOf(S.mode) + 1) % MODES.length]);
                if (placed && (ax || ay)) {
                    const pivot = oneToOne() ? model.localToWorld(halo.position.clone()) : placement.position.clone();
                    if (ax) rotateAbout(pivot, -ax * (oneToOne() ? 0.25 : 0.9) * dt);
                    if (ay && oneToOne()) placement.position.y -= ay * 0.1 * dt;
                    if (ay && S.mode === 'tabletop') {
                        S.scale = THREE.MathUtils.clamp(S.scale * Math.exp(-ay * 0.8 * dt), 1 / 40, 1 / 2);
                        placement.scale.setScalar(S.scale);
                        setScaleDependent();
                    }
                    adjusted(time);
                }
            } else if (src.handedness === 'left') {
                if (edge(1) && MEASURE && placed && oneToOne()) { S.measure = !S.measure; drawHelp(); }
                if (edge(4)) realign();
                if (edge(5)) { S.paused = !S.paused; drawHelp(); }
                if (edge(3)) { session.end(); return; }
                if (placed && (ax || ay)) {
                    const speed = oneToOne() ? 0.15 : 0.3;
                    const fwd = new THREE.Vector3(head.fwd.x, 0, head.fwd.z).normalize();
                    const right = new THREE.Vector3(-fwd.z, 0, fwd.x);
                    placement.position.addScaledVector(fwd, -ay * speed * dt).addScaledVector(right, ax * speed * dt);
                    adjusted(time);
                }
            }
        }
    }

    function updateVisibility() {
        const P = preset();
        placement.updateMatrixWorld(true);
        const cam = root.worldToLocal(head.pos.clone());  // head position in the z-up room frame
        const facing = walls.map(w => (cam.x - w.a[0]) * w.n[0] + (cam.y - w.a[1]) * w.n[1] > 0);
        twin.elements.forEach(el => {
            let vis;
            if (assetIds.has(el.id)) vis = P.asset !== 'hidden';
            else if (!P.room || el.group === 'ceiling') vis = false;
            else vis = el.host_wall === undefined || facing[el.host_wall];
            groups[el.id].visible = vis;
        });
        (S.cloud || []).forEach((seg, i) => { seg.visible = P.cloud && (i === 0 || (i >= 2 && facing[i - 2])); });
    }

    // ---------- frame loop ----------
    const lookTarget = new THREE.Vector3();
    renderer.setAnimationLoop((time, frame) => {
        if (!frame) return;
        const dt = S.lastTime ? Math.min((time - S.lastTime) / 1000, 0.1) : 0;
        S.lastTime = time;

        const viewer = frame.getViewerPose(refSpace);
        if (viewer) {
            head.pos.copy(viewer.transform.position);
            head.fwd.set(0, 0, -1).applyQuaternion(tmpQ.copy(viewer.transform.orientation));
            head.yaw = Math.atan2(-head.fwd.x, -head.fwd.z);
        }

        // Rays and aim points of every input source
        targets.clear();
        rays.clear();
        for (const src of session.inputSources) {
            const rp = frame.getPose(src.targetRaySpace, refSpace);
            if (!rp) continue;
            const origin = new THREE.Vector3().copy(rp.transform.position);
            const dir = new THREE.Vector3(0, 0, -1).applyQuaternion(tmpQ.copy(rp.transform.orientation));
            rays.set(src, { origin, dir });
            let point = null;
            const hs = hitSources.get(src);
            const results = hs ? frame.getHitTestResults(hs) : [];
            const hp = results.length ? results[0].getPose(refSpace) : null;
            if (hp) point = new THREE.Vector3().copy(hp.transform.position);
            else if (dir.y < -0.05) {
                const t = -origin.y / dir.y;
                if (t > 0 && t < 15) point = origin.clone().addScaledVector(dir, t);
            }
            if (point) targets.set(src, point);
        }
        const sources = [...session.inputSources];
        const active = sources.includes(S.activeSource) ? S.activeSource : sources.find(s => s.handedness === 'right') || sources[0];
        const aim = active && targets.get(active);
        const aiming = S.step === 'calibrate' || S.step === 'place';
        const touching = S.step === 'touch' || (S.measure && S.step === 'placed');
        reticle.visible = aiming && !!aim;
        if (reticle.visible) reticle.position.copy(aim);
        rayLines.forEach(({ c, line }) => {
            const p = c.userData.src && targets.get(c.userData.src);
            line.scale.z = touching ? TOUCH_OFFSET : aiming && p ? c.position.distanceTo(p) : 3;
        });
        // Touch balls: drawn exactly where a trigger pull records the point (the same ray data as onSelect)
        balls.forEach(b => { b.visible = false; });
        if (touching) rays.forEach((r, src) => {
            const ball = ballFor(src);
            ball.position.copy(r.origin).addScaledVector(r.dir, TOUCH_OFFSET);
            ball.visible = true;
        });
        if (S.step === 'auto') autoAlign(time, frame);

        // Tabletop: the model follows the aim point until it is put down
        if (S.mode === 'tabletop' && S.step === 'place') {
            placement.visible = !!aim;
            if (aim) { placement.position.copy(aim); placement.rotation.set(0, head.yaw, 0); }
        }

        handleGamepads(time, dt);
        if (S.ended) return;

        // 1:1 modes: follow the anchor (it corrects tracking drift); re-anchor after fine-tuning stops
        if (oneToOne() && S.step === 'placed') {
            if (S.anchor && frame.trackedAnchors?.has(S.anchor)) {
                const ap = frame.getPose(S.anchor.anchorSpace, refSpace);
                if (ap) { placement.position.copy(ap.transform.position); placement.quaternion.copy(ap.transform.orientation); }
            }
            if (S.reanchorAt && time > S.reanchorAt) { S.reanchorAt = 0; S.wantAnchor = true; }
            serveAnchor(frame);
        }

        if (S.running && !S.paused) loops.forEach(l => { l.offset += l.speed * dt; l.place(l.offset); });
        updateVisibility();

        // Panels: the status card hangs above the asset, facing the user; the help panel floats if no left controller
        const show = placement.visible && S.step === 'placed';
        card.mesh.visible = leader.visible = show;
        actions.mesh.visible = show && S.menu;
        if (show) {
            model.localToWorld(card.mesh.position.copy(cardAnchor));
            card.mesh.lookAt(lookTarget.set(head.pos.x, card.mesh.position.y, head.pos.z));
            const pos = leaderGeom.attributes.position;
            const bottom = card.mesh.position, sensorsAt = model.localToWorld(cardAnchor.clone().setY(markers[0].position.y));
            pos.setXYZ(0, bottom.x, bottom.y, bottom.z);
            pos.setXYZ(1, sensorsAt.x, sensorsAt.y, sensorsAt.z);
            pos.needsUpdate = true;
            if (actions.mesh.visible) {  // to the right of the status card, same baseline, facing the user
                const toHead = new THREE.Vector3(head.pos.x - bottom.x, 0, head.pos.z - bottom.z).normalize();
                const right = new THREE.Vector3(toHead.z, 0, -toHead.x);
                actions.mesh.position.copy(bottom).addScaledVector(right, oneToOne() ? 0.38 : 0.18);
                actions.mesh.lookAt(lookTarget.set(head.pos.x, actions.mesh.position.y, head.pos.z));
            }
        }
        if (info.mesh.visible) info.mesh.lookAt(lookTarget.set(head.pos.x, info.mesh.position.y, head.pos.z));
        if (help.mesh.parent === scene) {
            const right = new THREE.Vector3(-head.fwd.z, 0, head.fwd.x).normalize();
            const target = head.pos.clone().addScaledVector(head.fwd, 0.7).addScaledVector(right, -0.25).add(new THREE.Vector3(0, -0.35, 0));
            help.mesh.position.lerp(target, 0.08);
            help.mesh.lookAt(head.pos);
        }
        const helpKey = `${S.scale.toFixed(3)}`;
        if (helpKey !== S.helpKey) { S.helpKey = helpKey; drawHelp(); }

        renderer.render(scene, camera);

        // Frame-time samples (measurement sessions): one record every 10 s per view
        if (MEASURE && dt > 0) {
            S.frames.push(dt * 1000);
            if (!S.fpsSince) S.fpsSince = time;
            if (time - S.fpsSince > 10000) {
                const ft = [...S.frames].sort((a, b) => a - b);
                const q = (p) => ft[Math.min(ft.length - 1, Math.floor(p * ft.length))];
                logRecord('fps', {
                    mode: S.mode, view: preset().name, step: S.step, menu: S.menu, cloud: S.cloudState,
                    fps: (1000 * ft.length) / (time - S.fpsSince), ft_p50_ms: q(0.5), ft_p95_ms: q(0.95), ft_max_ms: ft[ft.length - 1],
                    calls: renderer.info.render.calls, triangles: renderer.info.render.triangles,
                });
                S.frames = [];
                S.fpsSince = time;
            }
        }
    });

    // ---------- start ----------
    const saved = readSaved();
    if (saved?.pos && saved?.quat) {
        S.onsitePose = { pos: new THREE.Vector3().fromArray(saved.pos), quat: new THREE.Quaternion().fromArray(saved.quat) };
        S.align = 'restored';  // X then starts a fresh automatic alignment
        S.alignInfo = saved.info
            || (typeof saved.calError === 'number' ? { text: `Corner distance error: ${Math.abs(saved.calError * 100).toFixed(1)} cm`, color: '#94a3b8' } : null);
        S.anchorUuid = saved.uuid || null;
        S.anchorStatus = 'Restored last alignment (pose)';
        if (saved.uuid && session.restorePersistentAnchor) {
            session.restorePersistentAnchor(saved.uuid).then(anchor => {
                if (S.ended) return;
                S.anchor = anchor;
                S.anchorStatus = 'Restored last alignment (persistent anchor)';
                drawHelp();
            }).catch(() => {
                S.anchorStatus = 'Saved anchor not found: check the overlay, press X to re-align if needed';
                drawHelp();
            });
        }
    }
    applyMode(mode);
    drawCard();

    session.addEventListener('end', () => {
        S.ended = true;
        renderer.setAnimationLoop(null);
        hitSources.forEach(hs => hs?.cancel());
        [card, info, help, actions].forEach(p => p.dispose());
        flushLog();
        scene.traverse(o => {
            o.geometry?.dispose();
            if (o.material && !o.isMesh) o.material.dispose?.();
        });
        Object.values(mats).forEach(m => m.dispose());
        xrayMats.forEach(m => m.dispose());
        [haloMat, reticle.material, ballMat, ...pins.map(p => p.material), ...markers.map(m => m.material)].forEach(m => m.dispose());
        renderer.dispose();
        renderer.domElement.remove();
        onEnd?.();
    });

    return {
        session,
        setSummary(next) { applySummary(next); },
        setRunning(next) { S.running = next; if (!S.ended) drawCard(); },
        end() { if (!S.ended) session.end().catch(() => {}); },
    };
}
