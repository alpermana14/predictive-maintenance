// Shared digital twin model code: used by the dashboard viewer (DigitalTwin.jsx) and the AR view (arSession.js),
// so both build exactly the same three.js objects from twin.json and cloud.bin.
import * as THREE from 'three';
import { mergeGeometries } from 'three/addons/utils/BufferGeometryUtils.js';

// Model (twin.json) and point cloud (cloud.bin) are static files served by this frontend's own nginx,
// exported from the Scan-to-BIM model (export_twin.py). No third-party service is involved.
export const TWIN_URL = `${import.meta.env.BASE_URL}twin/twin.json`;
export const CLOUD_URL = `${import.meta.env.BASE_URL}twin/cloud.bin`;
export const ZONE_COLORS = { A: '#22c55e', B: '#eab308', C: '#f97316', D: '#dc2626' };
export const NEUTRAL = '#3b82f6';
export const FLAGGED = '#f59e0b';

// Room frame is z-up (x east, y north); three.js is y-up
export const toThree = ([x, y, z]) => new THREE.Vector3(x, z, -y);

// ================= GEOMETRY =================
// Each part is a 2D profile in the plane (o, u, v) extruded by `depth` along u x v
function buildPart(part) {
    const shape = new THREE.Shape(part.shape.map(([a, b]) => new THREE.Vector2(a, b)));
    (part.holes || []).forEach(h => shape.holes.push(new THREE.Path(h.map(([a, b]) => new THREE.Vector2(a, b)))));
    const geom = new THREE.ExtrudeGeometry(shape, { depth: part.depth, bevelEnabled: false, curveSegments: 1 });
    const u = new THREE.Vector3(...part.u), v = new THREE.Vector3(...part.v);
    const n = new THREE.Vector3().crossVectors(u, v);
    geom.applyMatrix4(new THREE.Matrix4().makeBasis(u, v, n).setPosition(...part.o));
    return geom;
}

export function buildMaterials(twin) {
    const mats = {};
    Object.entries(twin.materials).forEach(([name, hex]) => {
        const opacity = twin.opacity[name] ?? 1;
        const glow = twin.emissive.includes(name);
        mats[name] = new THREE.MeshStandardMaterial({
            color: hex, roughness: 0.85, metalness: 0, side: THREE.DoubleSide,
            transparent: opacity < 1, opacity, depthWrite: opacity >= 1,
            emissive: glow ? hex : '#000000', emissiveIntensity: glow ? 0.55 : 0,
        });
    });
    return mats;
}

export function buildElement(el, mats) {
    const group = new THREE.Group();
    group.name = el.id;
    group.userData = { element: el };
    const byMat = {};
    el.parts.forEach(p => (byMat[p.m] = byMat[p.m] || []).push(buildPart(p)));
    Object.entries(byMat).forEach(([m, geoms]) => {
        const mesh = new THREE.Mesh(geoms.length > 1 ? mergeGeometries(geoms) : geoms[0], mats[m]);
        mesh.userData.elementId = el.id;
        group.add(mesh);
    });
    if (el.motion) {
        const loop = buildPlateLoop(el.motion, mats);
        [loop.mesh, ...loop.payloads].forEach(o => {
            o.userData.elementId = el.id;
            group.add(o);
        });
        group.userData.loop = loop;
    }
    return group;
}

// Moving conveyor plates: one InstancedMesh; place(offset) puts every plate on the loop at its arc length
// and turns it to face along the loop (the plate profile lies along +x). Payload boxes ride on the plates
// at a fixed arc position relative to them. A negative speed runs the loop clockwise.
function buildPlateLoop(m, mats) {
    const shape = new THREE.Shape(m.plate.map(([a, b]) => new THREE.Vector2(a, b)));
    const geom = new THREE.ExtrudeGeometry(shape, { depth: m.z1 - m.z0, bevelEnabled: false, curveSegments: 1 });
    const mesh = new THREE.InstancedMesh(geom, mats[m.material], m.count);
    const payloads = (m.payloads || []).map(p => {
        const [length, width, height] = p.size;
        const box = new THREE.Mesh(new THREE.BoxGeometry(length, width, height).translate(0, 0, m.z1 + height / 2), mats[p.material]);
        box.matrixAutoUpdate = false;
        box.userData.s = p.s;
        return box;
    });
    const straight = m.yn - m.ys, turn = Math.PI * m.radius, length = 2 * straight + 2 * turn;
    // east lane northwards, north turn, west lane southwards, south turn
    const at = (s) => {
        s = ((s % length) + length) % length;
        if (s < straight) return [m.xc + m.radius, m.ys + s, Math.PI / 2];
        s -= straight;
        if (s < turn) {
            const t = s / m.radius;
            return [m.xc + m.radius * Math.cos(t), m.yn + m.radius * Math.sin(t), t + Math.PI / 2];
        }
        s -= turn;
        if (s < straight) return [m.xc - m.radius, m.yn - s, -Math.PI / 2];
        const t = Math.PI + (s - straight) / m.radius;
        return [m.xc + m.radius * Math.cos(t), m.ys + m.radius * Math.sin(t), t + Math.PI / 2];
    };
    const matrix = new THREE.Matrix4(), rotation = new THREE.Matrix4();
    const place = (offset) => {
        for (let i = 0; i < m.count; i++) {
            const [x, y, heading] = at(offset + (i * length) / m.count);
            mesh.setMatrixAt(i, matrix.makeTranslation(x, y, m.z0).multiply(rotation.makeRotationZ(heading)));
        }
        mesh.instanceMatrix.needsUpdate = true;
        payloads.forEach(box => {
            const [x, y, heading] = at(offset + box.userData.s);
            box.matrix.makeTranslation(x, y, 0).multiply(rotation.makeRotationZ(heading));
            box.matrixWorldNeedsUpdate = true;
        });
    };
    place(0);
    mesh.computeBoundingSphere();  // the plates stay on the loop, so one bounding sphere is enough for picking
    return { mesh, payloads, place, speed: m.speed, offset: 0 };
}

// cloud.bin: 'PCL2', N, S, S segment counts (interior, ceiling, wall 0..n-1), min xyz, scale xyz,
// uint16 positions, uint8 RGB. Returns one Points object per segment so the cutaway can hide them.
export async function loadCloud(pointSize = 0.012) {
    const buf = await (await fetch(CLOUD_URL)).arrayBuffer();
    const view = new DataView(buf);
    const n = view.getUint32(4, true), nseg = view.getUint32(8, true);
    const counts = [...Array(nseg)].map((_, i) => view.getUint32(12 + 4 * i, true));
    const h = 12 + 4 * nseg;
    const mn = [0, 1, 2].map(i => view.getFloat32(h + 4 * i, true));
    const sc = [0, 1, 2].map(i => view.getFloat32(h + 12 + 4 * i, true));
    const q = new Uint16Array(buf, h + 24, n * 3);
    const rgb = new Uint8Array(buf, h + 24 + n * 6, n * 3);
    const pos = new Float32Array(n * 3), col = new Float32Array(n * 3);
    for (let i = 0; i < n; i++) {
        pos[3 * i] = mn[0] + q[3 * i] * sc[0];
        pos[3 * i + 1] = mn[1] + q[3 * i + 1] * sc[1];
        pos[3 * i + 2] = mn[2] + q[3 * i + 2] * sc[2];
        for (let k = 0; k < 3; k++) col[3 * i + k] = Math.pow(rgb[3 * i + k] / 255, 2.2); // sRGB -> linear
    }
    const posAttr = new THREE.BufferAttribute(pos, 3), colAttr = new THREE.BufferAttribute(col, 3);
    const material = new THREE.PointsMaterial({ size: pointSize, vertexColors: true });
    let start = 0;
    return counts.map(count => {
        const geom = new THREE.BufferGeometry();
        geom.setAttribute('position', posAttr);
        geom.setAttribute('color', colAttr);
        geom.setDrawRange(start, count);
        start += count;
        return new THREE.Points(geom, material);
    });
}

// Wall planes of the room outline, for the cutaway: n is the inward normal of the wall from a to b
export function wallPlanes(twin) {
    const poly = twin.room.polygon;
    return poly.map((a, i) => {
        const b = poly[(i + 1) % poly.length];
        const len = Math.hypot(b[0] - a[0], b[1] - a[1]);
        return { a, n: [-(b[1] - a[1]) / len, (b[0] - a[0]) / len] };
    });
}

// ================= SENSOR STATUS =================
// Colours follow the backend: ISO zone of /api/summary for vibration, amber for channels the
// distribution-shift episode currently flags, blue otherwise. No thresholds are re-implemented here.
export function sensorColor(sensor, summary) {
    const keys = sensor.metrics.map(m => m.key);
    const flagged = summary?.shift_episode?.currently_flagged || [];
    if (keys.includes('z_rms') && summary?.iso_zone) return ZONE_COLORS[summary.iso_zone] || NEUTRAL;
    if (keys.some(k => flagged.includes(k))) return FLAGGED;
    return NEUTRAL;
}

export const fmt = (v) => (typeof v === 'number' ? v.toFixed(2) : '–');
