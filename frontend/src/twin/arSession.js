// Mixed-reality view of the digital twin for Meta Quest 3 (WebXR 'immersive-ar' in the Meta Quest Browser).
//
// Two modes:
//   onsite   : the twin at 1:1, aligned to the real lab by aiming at two floor corners. The alignment is kept as a
//              persistent WebXR anchor (and in localStorage), so it only has to be done once per headset.
//   tabletop : a scale model of the lab placed on a table or the floor.
//
// The live machine state comes from the same /api/summary object the dashboard already polls (setSummary).
// In-headset UI is drawn on canvas textures, because DOM overlays are not available in immersive mode on Quest.
import * as THREE from 'three';
import { ZONE_COLORS, NEUTRAL, FLAGGED, toThree, buildMaterials, buildElement, loadCloud, wallPlanes, sensorColor, fmt } from './twinModel';

const STORAGE_KEY = 'pm-twin-ar-calibration-v1';
const CLOUD_POINT_SIZE = 0.012;
const Y_AXIS = new THREE.Vector3(0, 1, 0);

// Alignment: two floor corners of the room outline (indices into twin.room.polygon) that no furniture blocks.
// 0 = south-west corner (the room origin), 5 = north-west corner; they are 7.2 m apart along the west wall.
export const CALIBRATION = [
    { corner: 0, label: 'Corner 1: SOUTH-WEST', hint: 'The floor corner on the conveyor / air-conditioner wall, at the end FAR from the conveyor.' },
    { corner: 5, label: 'Corner 2: NORTH-WEST', hint: 'From corner 1, follow the other wall to its far end: the floor corner on the interactive display (TV) wall.' },
];

// What each view shows. asset: the conveyor and motor ('solid' | 'xray' | 'hidden'); room: the rest of the model.
const PRESETS = {
    onsite: [
        { name: 'Status overlay', asset: 'hidden', room: false, cloud: false },
        { name: 'X-ray conveyor', asset: 'xray', room: false, cloud: false },
        { name: 'Scan check (point cloud)', asset: 'xray', room: false, cloud: true },
        { name: 'Full twin', asset: 'solid', room: true, cloud: false },
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
    // lines: [{ text, size, color, bold, gap }]
    const draw = (lines, accent = '#5eead4') => {
        const pad = 28;
        const rows = [];
        let y = pad;
        lines.forEach(l => {
            const size = l.size || 30;
            const font = `${l.bold ? '700' : '400'} ${size}px system-ui, -apple-system, Segoe UI, Roboto, sans-serif`;
            g.font = font;
            wrap(g, l.text, pxW - 2 * pad - 12).forEach(t => {
                y += size * 1.28;
                rows.push({ t, y, font, color: l.color || '#e2e8f0' });
            });
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
        rows.forEach(r => {
            g.font = r.font;
            g.fillStyle = r.color;
            g.fillText(r.t, pad + 12, r.y - 6);
        });
        tex.needsUpdate = true;
        layout();
    };
    const setWidth = (nw) => { w = nw; layout(); };
    const dispose = () => { mesh.geometry.dispose(); mesh.material.dispose(); tex.dispose(); };
    return { mesh, draw, setWidth, dispose };
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
    scene.add(card.mesh, info.mesh, help.mesh);
    info.mesh.visible = false;

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

    // ---------- session (requested while the click's user activation is still valid) ----------
    const session = await navigator.xr.requestSession('immersive-ar', {
        requiredFeatures: ['local-floor'],
        optionalFeatures: ['hit-test', 'anchors', 'hand-tracking'],
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
    const S = {
        mode, preset: 0, step: 'place', calPoints: [], calError: null, scale: 1 / 12,
        summary, running, paused: false, ended: false,
        onsitePose: null, anchor: null, anchorUuid: null, anchorStatus: '', wantAnchor: false, reanchorAt: 0,
        cloud: null, cloudState: 'idle', message: '', activeSource: null, lastTime: 0, helpKey: '',
    };
    const hitSources = new Map(), targets = new Map(), rays = new Map(), prevButtons = new Map();
    const head = { pos: new THREE.Vector3(0, 1.6, 0), fwd: new THREE.Vector3(0, 0, -1), yaw: 0 };

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
                uuid: S.anchorUuid, calError: S.calError, savedAt: new Date().toISOString(),
            }));
        } catch { /* storage blocked: the alignment then lasts for this session only */ }
    };

    function setScaleDependent() {
        const s = placement.scale.x;
        markers.forEach(m => m.scale.setScalar((S.mode === 'onsite' ? 0.025 : 0.006) / (0.03 * s)));
        if (S.cloud) S.cloud[0].material.size = CLOUD_POINT_SIZE * s;
        card.setWidth(S.mode === 'onsite' ? 0.36 : 0.16);
        info.setWidth(S.mode === 'onsite' ? 0.3 : 0.18);
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

    function startCalibration() {
        S.step = 'calibrate';
        S.calPoints = [];
        S.anchor = null;
        placement.visible = false;
        pins.forEach(p => { p.visible = false; });
        info.mesh.visible = false;
        drawHelp();
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
            } else startCalibration();
        }
        setScaleDependent();
        applyPreset();
    }

    // Two measured floor points p1, p2 (world) <-> the same corners q1, q2 in the model: yaw + translation, no scale.
    // The difference between the measured and the modelled corner distance is reported as a quality check.
    function solveCalibration() {
        const [p1, p2] = S.calPoints;
        const [q1, q2] = CALIBRATION.map(c => toThree([...poly[c.corner], 0]));
        const theta = Math.atan2(q2.z - q1.z, q2.x - q1.x) - Math.atan2(p2.z - p1.z, p2.x - p1.x);
        const quat = new THREE.Quaternion().setFromAxisAngle(Y_AXIS, theta);
        const r1 = q1.clone().applyQuaternion(quat), r2 = q2.clone().applyQuaternion(quat);
        placement.position.set((p1.x - r1.x + p2.x - r2.x) / 2, (p1.y + p2.y) / 2, (p1.z - r1.z + p2.z - r2.z) / 2);
        placement.quaternion.copy(quat);
        S.calError = Math.hypot(p2.x - p1.x, p2.z - p1.z) - Math.hypot(q2.x - q1.x, q2.z - q1.z);
        S.onsitePose = { pos: placement.position.clone(), quat: placement.quaternion.clone() };
        placement.visible = true;
        pins.forEach(p => { p.visible = false; });
        S.step = 'placed';
        S.preset = 1;  // x-ray first, so the alignment can be checked against the real conveyor
        applyPreset();
        S.wantAnchor = true;
        flash(Math.abs(S.calError) > 0.1
            ? 'Corner distance is off by more than 10 cm: check the corners and press X to align again.'
            : 'Aligned. The x-ray conveyor should sit on the real one; fine-tune with the thumbsticks.', 7000);
    }

    function onSelect(src) {
        S.activeSource = src;
        const point = targets.get(src);
        if (S.step === 'calibrate') {
            if (!point) return flash('No surface found: aim the ray at the floor corner.');
            pins[S.calPoints.length].position.copy(point);
            pins[S.calPoints.length].visible = true;
            S.calPoints.push(point.clone());
            if (S.calPoints.length === 2) solveCalibration(); else drawHelp();
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
            { text: `Alert level: ${s?.machine_status ?? '–'}`, size: 28, color: '#cbd5e1', gap: 10 },
            ...twin.sensors.flatMap(x => x.metrics).map(m => {
                const warn = flagged.includes(m.key);
                return { text: `${m.label}: ${fmt(s?.metrics?.[m.key])} ${m.unit}${warn ? '  (shift flagged)' : ''}`, size: 32, color: warn ? FLAGGED : '#f1f5f9' };
            }),
            { text: `Reading: ${s?.reading_time || s?.data_timestamp || 'waiting for data'}`, size: 22, color: '#94a3b8' },
        ];
        if (!S.running) lines.push({ text: 'No live data: the sensor feed is stale', size: 22, color: FLAGGED });
        card.draw(lines, zone);
        drawInfo();
    }

    function drawHelp() {
        const L = [];
        const P = preset();
        L.push({ text: S.mode === 'onsite' ? 'ON-SITE  ·  1:1' : `TABLETOP  ·  1:${Math.round(1 / S.scale)}`, bold: true, size: 40, color: '#5eead4', gap: 6 });
        if (S.step === 'calibrate') {
            const c = CALIBRATION[S.calPoints.length];
            L.push({ text: `Align, step ${S.calPoints.length + 1} of 2`, bold: true, size: 32 });
            L.push({ text: c.label, bold: true, size: 32, color: '#fde047' });
            L.push({ text: c.hint, size: 28, color: '#cbd5e1' });
            L.push({ text: 'Aim the ray exactly at that corner on the floor and pull the TRIGGER.', size: 28, gap: 10 });
        } else if (S.step === 'place') {
            L.push({ text: 'Aim at a table or the floor. Pull the TRIGGER to put the model down.', size: 30, gap: 10 });
        } else {
            L.push({ text: `View ${S.preset + 1}/${PRESETS[S.mode].length}: ${P.name}`, bold: true, size: 32 });
            if (P.cloud && S.cloudState !== 'ready') L.push({ text: S.cloudState === 'error' ? 'Point cloud failed to load' : 'Loading point cloud (5 MB)...', size: 26, color: '#fde047' });
            if (S.mode === 'onsite') {
                if (S.calError !== null) {
                    const cm = Math.abs(S.calError * 100);
                    L.push({ text: `Corner distance error: ${cm.toFixed(1)} cm`, size: 26, color: cm <= 5 ? '#86efac' : cm <= 10 ? '#fde047' : '#fca5a5' });
                }
                if (S.anchorStatus) L.push({ text: S.anchorStatus, size: 26, color: '#94a3b8' });
            }
            L.push({ text: 'TRIGGER on an object or sensor: details', size: 28, gap: 10 });
        }
        if (S.message) L.push({ text: S.message, size: 28, color: '#fca5a5', gap: 10 });
        L.push({ text: 'RIGHT   A: next view   B: on-site / tabletop', size: 26, color: '#e2e8f0' });
        L.push({ text: `             stick: rotate  ·  ${S.mode === 'onsite' ? 'up / down' : 'bigger / smaller'}`, size: 26, color: '#e2e8f0' });
        L.push({ text: `LEFT     X: ${S.mode === 'onsite' ? 're-align' : 're-place'}   Y: ${S.paused ? 'run' : 'pause'} animation`, size: 26, color: '#e2e8f0' });
        L.push({ text: '             stick: move   ·   click stick: exit AR', size: 26, color: '#e2e8f0' });
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
        if (S.mode !== 'onsite') return;
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
                if (edge(4) && placed) { S.preset = (S.preset + 1) % PRESETS[S.mode].length; applyPreset(); }
                if (edge(5)) applyMode(S.mode === 'onsite' ? 'tabletop' : 'onsite');
                if (placed && (ax || ay)) {
                    const pivot = S.mode === 'onsite' ? model.localToWorld(halo.position.clone()) : placement.position.clone();
                    if (ax) rotateAbout(pivot, -ax * (S.mode === 'onsite' ? 0.25 : 0.9) * dt);
                    if (ay && S.mode === 'onsite') placement.position.y -= ay * 0.1 * dt;
                    if (ay && S.mode === 'tabletop') {
                        S.scale = THREE.MathUtils.clamp(S.scale * Math.exp(-ay * 0.8 * dt), 1 / 40, 1 / 2);
                        placement.scale.setScalar(S.scale);
                        setScaleDependent();
                    }
                    adjusted(time);
                }
            } else if (src.handedness === 'left') {
                if (edge(4)) { if (S.mode === 'onsite') startCalibration(); else { S.step = 'place'; drawHelp(); } }
                if (edge(5)) { S.paused = !S.paused; drawHelp(); }
                if (edge(3)) { session.end(); return; }
                if (placed && (ax || ay)) {
                    const speed = S.mode === 'onsite' ? 0.15 : 0.3;
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
        reticle.visible = aiming && !!aim;
        if (reticle.visible) reticle.position.copy(aim);
        rayLines.forEach(({ c, line }) => {
            const p = c.userData.src && targets.get(c.userData.src);
            line.scale.z = aiming && p ? c.position.distanceTo(p) : 3;
        });

        // Tabletop: the model follows the aim point until it is put down
        if (S.mode === 'tabletop' && S.step === 'place') {
            placement.visible = !!aim;
            if (aim) { placement.position.copy(aim); placement.rotation.set(0, head.yaw, 0); }
        }

        handleGamepads(time, dt);
        if (S.ended) return;

        // On-site: follow the anchor (it corrects tracking drift); re-anchor after fine-tuning stops
        if (S.mode === 'onsite' && S.step === 'placed') {
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
        if (show) {
            model.localToWorld(card.mesh.position.copy(cardAnchor));
            card.mesh.lookAt(lookTarget.set(head.pos.x, card.mesh.position.y, head.pos.z));
            const pos = leaderGeom.attributes.position;
            const bottom = card.mesh.position, sensorsAt = model.localToWorld(cardAnchor.clone().setY(markers[0].position.y));
            pos.setXYZ(0, bottom.x, bottom.y, bottom.z);
            pos.setXYZ(1, sensorsAt.x, sensorsAt.y, sensorsAt.z);
            pos.needsUpdate = true;
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
    });

    // ---------- start ----------
    const saved = readSaved();
    if (saved?.pos && saved?.quat) {
        S.onsitePose = { pos: new THREE.Vector3().fromArray(saved.pos), quat: new THREE.Quaternion().fromArray(saved.quat) };
        S.calError = saved.calError ?? null;
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
        [card, info, help].forEach(p => p.dispose());
        scene.traverse(o => {
            o.geometry?.dispose();
            if (o.material && !o.isMesh) o.material.dispose?.();
        });
        Object.values(mats).forEach(m => m.dispose());
        xrayMats.forEach(m => m.dispose());
        [haloMat, reticle.material, ...pins.map(p => p.material), ...markers.map(m => m.material)].forEach(m => m.dispose());
        renderer.dispose();
        renderer.domElement.remove();
        onEnd?.();
    });

    return {
        session,
        setSummary(next) { S.summary = next; if (!S.ended) drawCard(); },
        setRunning(next) { S.running = next; if (!S.ended) drawCard(); },
        end() { if (!S.ended) session.end().catch(() => {}); },
    };
}
