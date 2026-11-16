import React, { useEffect, useRef, useState } from 'react';
import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { CSS2DRenderer, CSS2DObject } from 'three/addons/renderers/CSS2DRenderer.js';
import { TWIN_URL, ZONE_COLORS, NEUTRAL, FLAGGED, toThree, buildMaterials, buildElement, loadCloud, wallPlanes, sensorColor, fmt } from '../twin/twinModel';
import ARLaunch from './ARLaunch';

// ================= 1. CONFIGURATION =================
// Geometry, point cloud loading and sensor colours live in ../twin/twinModel.js (shared with the AR view)
const VIEWS = {
    room: { pos: [9.2, -4.4, 6.2], target: [2.8, 2.6, 0.5] },
    asset: { pos: [3.3, 1.7, 1.45], target: [4.85, 0.6, 0.75] },
};

// ================= 2. COMPONENT =================
// running: live sensor data is arriving (the dashboard's stale check), so the conveyor is animated
export default function DigitalTwin({ summary, selectedTarget, onSelectTarget, running = true }) {
    const mountRef = useRef(null);
    const ctx = useRef({});
    const [twin, setTwin] = useState(null);
    const [error, setError] = useState(null);
    const [selected, setSelected] = useState(null);
    const [layers, setLayers] = useState({ ceiling: false, walls: true, cloud: false });
    const [paused, setPaused] = useState(false);
    const [cloudState, setCloudState] = useState('idle');

    useEffect(() => {
        fetch(TWIN_URL).then(r => r.json()).then(setTwin).catch(e => setError(String(e)));
    }, []);

    // Scene setup (once the model is loaded)
    useEffect(() => {
        if (!twin) return;
        const mount = mountRef.current;
        const renderer = new THREE.WebGLRenderer({ antialias: true });
        renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
        mount.appendChild(renderer.domElement);
        const labels = new CSS2DRenderer();
        Object.assign(labels.domElement.style, { position: 'absolute', top: '0', left: '0', pointerEvents: 'none' });
        mount.appendChild(labels.domElement);

        const scene = new THREE.Scene();
        scene.background = new THREE.Color('#eef2f7');
        scene.add(new THREE.HemisphereLight('#ffffff', '#8890a0', 2.2));
        const sun = new THREE.DirectionalLight('#ffffff', 1.4);
        sun.position.set(6, 10, 4);
        scene.add(sun);

        const camera = new THREE.PerspectiveCamera(45, 1, 0.05, 100);
        const controls = new OrbitControls(camera, renderer.domElement);
        controls.enableDamping = true;
        controls.maxPolarAngle = Math.PI * 0.495;

        // Model in the room frame, rotated to y-up
        const root = new THREE.Group();
        root.rotation.x = -Math.PI / 2;
        scene.add(root);
        const mats = buildMaterials(twin);
        const groups = {};
        twin.elements.forEach(el => {
            const g = buildElement(el, mats);
            groups[el.id] = g;
            root.add(g);
        });

        // Asset status box around the monitored conveyor
        root.updateMatrixWorld(true);
        const assetBox = new THREE.Box3().setFromObject(groups[twin.asset]).expandByScalar(0.04);
        const size = assetBox.getSize(new THREE.Vector3()), center = assetBox.getCenter(new THREE.Vector3());
        const haloMat = new THREE.MeshBasicMaterial({ color: NEUTRAL, transparent: true, opacity: 0.12, depthWrite: false });
        const halo = new THREE.Mesh(new THREE.BoxGeometry(size.x, size.y, size.z), haloMat);
        halo.position.copy(center);
        const haloEdges = new THREE.LineSegments(new THREE.EdgesGeometry(halo.geometry), new THREE.LineBasicMaterial({ color: NEUTRAL }));
        halo.add(haloEdges);
        scene.add(halo);

        // Sensor markers (the sensors sit close together on the drive motor, so one shared label is used)
        const sensors = twin.sensors.map(s => {
            const marker = new THREE.Mesh(new THREE.SphereGeometry(0.03, 16, 12), new THREE.MeshBasicMaterial({ color: NEUTRAL }));
            marker.position.copy(toThree(s.pos));
            marker.userData.sensor = s;
            scene.add(marker);
            return { sensor: s, marker };
        });

        // Live label to the left of the sensors: one row per metric, clicking a row selects it in the charts
        const anchor = document.createElement('div');
        const card = document.createElement('div');
        card.className = 'bg-white/95 border border-gray-200 rounded-md shadow px-2 py-1 text-[11px] leading-tight';
        Object.assign(card.style, { pointerEvents: 'auto', transform: 'translateX(calc(-50% - 28px))' });
        anchor.appendChild(card);
        const cardTitle = document.createElement('div');
        cardTitle.className = 'font-semibold text-gray-800 mb-0.5';
        card.appendChild(cardTitle);
        const rows = twin.sensors.flatMap(s => s.metrics).map(m => {
            const b = document.createElement('button');
            b.onclick = (ev) => { ev.stopPropagation(); ctx.current.onSelectTarget?.(m.key); };
            card.appendChild(b);
            return { metric: m, el: b };
        });
        const cardObj = new CSS2DObject(anchor);
        sensors.forEach(({ marker }) => cardObj.position.add(marker.position));
        cardObj.position.divideScalar(sensors.length);
        scene.add(cardObj);

        // Selection outline
        const selBox = new THREE.BoxHelper(undefined, '#facc15');
        selBox.visible = false;
        scene.add(selBox);

        // Click to select (ignored when the pointer was dragged to orbit)
        const raycaster = new THREE.Raycaster();
        let down = null;
        const onDown = (e) => { down = [e.clientX, e.clientY]; };
        const onUp = (e) => {
            if (!down || Math.hypot(e.clientX - down[0], e.clientY - down[1]) > 4) return;
            const r = renderer.domElement.getBoundingClientRect();
            raycaster.setFromCamera(new THREE.Vector2(((e.clientX - r.left) / r.width) * 2 - 1, -((e.clientY - r.top) / r.height) * 2 + 1), camera);
            const pickable = [...sensors.map(s => s.marker), ...Object.values(groups).filter(g => g.visible)];
            const hit = raycaster.intersectObjects(pickable, true).find(h => h.object.visible && h.object.material?.opacity !== 0);
            if (!hit) { setSelected(null); selBox.visible = false; return; }
            if (hit.object.userData.sensor) {
                const s = hit.object.userData.sensor;
                ctx.current.onSelectTarget?.(s.metrics[0].key);
                setSelected(groups[s.element].userData.element);
                selBox.setFromObject(groups[s.element]);
            } else {
                const id = hit.object.userData.elementId;
                setSelected(groups[id].userData.element);
                selBox.setFromObject(groups[id]);
            }
            selBox.visible = true;
        };
        renderer.domElement.addEventListener('pointerdown', onDown);
        renderer.domElement.addEventListener('pointerup', onUp);

        // Walls between the camera and the room are hidden (cutaway), together with what hangs on them;
        // the ceiling slab is hidden while the camera is above it. Point cloud segments follow the same rule.
        const ceilingZ = twin.room.ceiling_height;
        const walls = wallPlanes(twin);
        const updateVisibility = () => {
            const L = ctx.current.layers || { ceiling: false, walls: true, cloud: false };
            const camLocal = [camera.position.x, -camera.position.z];
            const facing = walls.map(w => (camLocal[0] - w.a[0]) * w.n[0] + (camLocal[1] - w.a[1]) * w.n[1] > 0);
            const below = camera.position.y < ceilingZ;
            twin.elements.forEach(el => {
                let vis = true;
                if (el.group === 'ceiling') vis = L.ceiling && (el.id !== twin.ceiling || below);
                if (el.host_wall !== undefined) vis = L.walls && facing[el.host_wall];
                groups[el.id].visible = vis;
            });
            (ctx.current.cloud || []).forEach((seg, i) => {
                seg.visible = L.cloud && (i === 0 || (i === 1 ? below : L.walls && facing[i - 2]));
            });
        };

        const setView = (name) => {
            const v = VIEWS[name];
            camera.position.copy(toThree(v.pos));
            controls.target.copy(toThree(v.target));
            controls.update();
        };
        setView('room');

        const resize = () => {
            const w = mount.clientWidth, h = mount.clientHeight;
            renderer.setSize(w, h);
            labels.setSize(w, h);
            camera.aspect = w / h;
            camera.updateProjectionMatrix();
        };
        const ro = new ResizeObserver(resize);
        ro.observe(mount);
        resize();

        const loops = Object.values(groups).map(g => g.userData.loop).filter(Boolean);
        const clock = new THREE.Clock();
        let raf;
        const tick = () => {
            const dt = Math.min(clock.getDelta(), 0.1);  // no jump after the tab was in the background
            if (ctx.current.moving) {
                loops.forEach(l => { l.offset += l.speed * dt; l.place(l.offset); });
            }
            controls.update();
            updateVisibility();
            renderer.render(scene, camera);
            labels.render(scene, camera);
            raf = requestAnimationFrame(tick);
        };
        tick();

        const c = ctx.current;
        Object.assign(c, { root, sensors, rows, cardTitle, haloMat, haloEdges, setView, cloud: null });
        return () => {
            cancelAnimationFrame(raf);
            ro.disconnect();
            renderer.domElement.removeEventListener('pointerdown', onDown);
            renderer.domElement.removeEventListener('pointerup', onUp);
            controls.dispose();
            scene.traverse(o => { o.geometry?.dispose(); });
            Object.values(mats).forEach(m => m.dispose());
            renderer.dispose();
            mount.removeChild(renderer.domElement);
            mount.removeChild(labels.domElement);
            Object.assign(c, { root: null, sensors: null, rows: null, cloud: null, setView: null });
        };
    }, [twin]);

    // Keep the latest props/layers available to the render loop and DOM handlers
    useEffect(() => { ctx.current.onSelectTarget = onSelectTarget; }, [onSelectTarget]);
    useEffect(() => { ctx.current.layers = layers; }, [layers]);
    useEffect(() => { ctx.current.moving = running && !paused; }, [running, paused]);

    // Live values from /api/summary
    useEffect(() => {
        const c = ctx.current;
        if (!c.sensors || !summary) return;
        const zoneColor = ZONE_COLORS[summary.iso_zone] || NEUTRAL;
        c.haloMat.color.set(zoneColor);
        c.haloEdges.material.color.set(zoneColor);
        const flagged = summary.shift_episode?.currently_flagged || [];
        c.sensors.forEach(({ sensor, marker }) => marker.material.color.set(sensorColor(sensor, summary)));
        c.cardTitle.textContent = `${twin.asset} · Zone ${summary.iso_zone} · ${summary.machine_status ?? ''}`;
        c.rows.forEach(({ metric, el }) => {
            const warn = flagged.includes(metric.key) ? ' ⚠' : '';
            el.textContent = `${metric.label}: ${fmt(summary.metrics?.[metric.key])} ${metric.unit}${warn}`;
            el.className = `block w-full text-left px-1 rounded font-mono ${metric.key === selectedTarget ? 'bg-blue-600 text-white' : 'hover:bg-blue-50 text-gray-700'}`;
        });
    }, [summary, selectedTarget, twin]);

    const toggle = (k) => {
        setLayers(l => ({ ...l, [k]: !l[k] }));
        // Point cloud is loaded on first use (5 MB)
        if (k === 'cloud' && cloudState === 'idle' && ctx.current.root) {
            setCloudState('loading');
            loadCloud().then(segments => {
                if (!ctx.current.root) return;
                ctx.current.cloud = segments;
                segments.forEach(s => ctx.current.root.add(s));
                setCloudState('ready');
            }).catch(() => setCloudState('error'));
        }
    };
    const isAsset = selected && twin && (selected.id === twin.asset || twin.sensors.some(s => s.element === selected.id));
    const btn = (on) => `px-3 py-1.5 rounded-md text-xs font-semibold border transition-colors ${on ? 'bg-blue-600 text-white border-blue-600' : 'bg-white text-gray-700 border-gray-200 hover:bg-gray-50'}`;

    return (
        <div className="bg-white p-6 rounded-xl shadow-sm border border-gray-100 mb-6">
            <div className="flex flex-col md:flex-row md:items-center justify-between gap-3 mb-4">
                <h2 className="text-lg font-semibold flex items-center">
                    <span className="w-2 h-2 rounded-full bg-teal-500 mr-2"></span>Digital Twin: {twin?.room.name || 'loading...'}
                </h2>
                <div className="flex flex-wrap gap-2">
                    <button className={btn(layers.walls)} onClick={() => toggle('walls')}>Walls</button>
                    <button className={btn(layers.ceiling)} onClick={() => toggle('ceiling')}>Ceiling & lights</button>
                    <button className={btn(layers.cloud)} onClick={() => toggle('cloud')}>
                        Point cloud{cloudState === 'loading' ? ' (loading...)' : cloudState === 'error' ? ' (failed)' : ''}
                    </button>
                    <button className={btn(false)} onClick={() => ctx.current.setView?.('asset')}>Focus motor & conveyor</button>
                    <button className={btn(false)} onClick={() => ctx.current.setView?.('room')}>Whole room</button>
                    <button className={btn(false)} onClick={() => setPaused(p => !p)}>{paused ? '▶ Run conveyor' : '⏸ Pause conveyor'}</button>
                    <ARLaunch twin={twin} summary={summary} running={running && !paused} />
                </div>
            </div>

            <div className="relative h-[560px] rounded-lg overflow-hidden border border-gray-100" ref={mountRef}>
                {error && <div className="absolute inset-0 flex items-center justify-center text-red-600 text-sm">Model could not be loaded: {error}</div>}

                {/* Selected element */}
                {selected && (
                    <div className="absolute top-3 left-3 w-72 bg-white/95 rounded-lg shadow-lg border border-gray-200 p-3 text-sm">
                        <div className="flex justify-between items-start">
                            <div>
                                <div className="font-semibold text-gray-900">{selected.name}</div>
                                <div className="text-xs text-gray-500">{selected.category} · Mark {selected.mark}</div>
                            </div>
                            <button className="text-gray-400 hover:text-gray-700" onClick={() => setSelected(null)}>✕</button>
                        </div>
                        {selected.note && <p className="text-xs text-gray-600 mt-2">{selected.note}</p>}
                        {isAsset && summary && (
                            <div className="mt-3 border-t pt-2">
                                <div className="flex items-center gap-2 text-xs font-semibold">
                                    <span className="w-3 h-3 rounded-sm" style={{ background: ZONE_COLORS[summary.iso_zone] || NEUTRAL }}></span>
                                    Zone {summary.iso_zone} - {summary.status} · Alert level: {summary.machine_status}
                                </div>
                                <div className="grid grid-cols-2 gap-1 mt-2">
                                    {twin.sensors.flatMap(s => s.metrics).map(m => (
                                        <button key={m.key} onClick={() => onSelectTarget(m.key)}
                                            className={`text-left px-2 py-1 rounded text-xs font-mono ${m.key === selectedTarget ? 'bg-blue-600 text-white' : 'bg-gray-50 hover:bg-blue-50'}`}>
                                            {m.label}: {fmt(summary.metrics?.[m.key])} {m.unit}
                                        </button>
                                    ))}
                                </div>
                                <div className="text-[10px] text-gray-400 mt-2">Reading: {summary.reading_time || summary.data_timestamp}</div>
                            </div>
                        )}
                    </div>
                )}

                {/* Legend */}
                <div className="absolute bottom-3 left-3 bg-white/90 rounded-md border border-gray-200 px-2 py-1.5 text-[10px] text-gray-600 flex flex-wrap gap-x-3 gap-y-1">
                    {Object.entries(ZONE_COLORS).map(([z, c]) => (
                        <span key={z} className="flex items-center gap-1"><span className="w-2.5 h-2.5 rounded-sm" style={{ background: c }}></span>Zone {z}</span>
                    ))}
                    <span className="flex items-center gap-1"><span className="w-2.5 h-2.5 rounded-sm" style={{ background: FLAGGED }}></span>Shift flagged</span>
                    <span className="font-semibold">
                        Conveyor: {!running ? 'stopped (no live data)' : paused ? 'paused' : 'moving (simulation)'}
                    </span>
                    <span className="text-gray-400">Click an object for details · click a sensor value to chart it</span>
                </div>
            </div>
        </div>
    );
}
