import React, { useEffect, useRef, useState } from 'react';
import { startTwinAR, clearSavedAlignment } from '../twin/arSession';

// AR buttons for the digital twin card. They are active in a WebXR browser with passthrough AR
// (Meta Quest Browser on Quest 3); elsewhere they stay disabled and say why.
const initialSupport = () => {
    if (typeof window === 'undefined') return 'no';
    if (!window.isSecureContext) return 'insecure';
    return navigator.xr ? 'checking' : 'no';
};

export default function ARLaunch({ twin, summary, running }) {
    const [support, setSupport] = useState(initialSupport);
    const [active, setActive] = useState(false);
    const [error, setError] = useState(null);
    const [cleared, setCleared] = useState(false);
    const ar = useRef(null);
    const latest = useRef({ summary, running });

    useEffect(() => {
        if (support !== 'checking') return;
        navigator.xr.isSessionSupported('immersive-ar')
            .then(ok => setSupport(ok ? 'yes' : 'no'))
            .catch(() => setSupport('no'));
    }, [support]);

    // Live data from the dashboard's /api/summary polling goes straight into the headset view
    useEffect(() => {
        latest.current = { summary, running };
        ar.current?.setSummary(summary);
    }, [summary, running]);
    useEffect(() => { ar.current?.setRunning(running); }, [running]);
    useEffect(() => {
        const handle = ar;
        return () => handle.current?.end();
    }, []);

    if (!twin) return null;

    const start = (mode) => {
        setError(null);
        setCleared(false);
        startTwinAR({ twin, mode, summary, running, onEnd: () => { ar.current = null; setActive(false); } })
            .then(h => {
                ar.current = h;
                h.setSummary(latest.current.summary);
                h.setRunning(latest.current.running);
                setActive(true);
            })
            .catch(e => setError(e?.message || String(e)));
    };

    const enabled = support === 'yes' && !active;
    const reason = support === 'insecure' ? 'AR needs the dashboard to be opened over HTTPS'
        : support === 'yes' ? '' : 'Open this dashboard in the Meta Quest Browser on the headset to use AR';
    const cls = `px-3 py-1.5 rounded-md text-xs font-semibold border transition-colors ${enabled
        ? 'bg-teal-600 text-white border-teal-600 hover:bg-teal-700'
        : 'bg-gray-50 text-gray-400 border-gray-200 cursor-not-allowed'}`;

    return (
        <>
            <button className={cls} disabled={!enabled} onClick={() => start('onsite')}
                title={reason || 'Overlay the twin 1:1 on the real lab (aligned automatically to the headset\'s room scan)'}>
                AR on-site
            </button>
            <button className={cls} disabled={!enabled} onClick={() => start('conveyor')}
                title={reason || 'Only the conveyor at 1:1 (align by touching two corners of its frame with the controller)'}>
                AR conveyor
            </button>
            <button className={cls} disabled={!enabled} onClick={() => start('tabletop')}
                title={reason || 'Place a scale model of the lab on a table'}>
                AR tabletop
            </button>
            {support === 'yes' && !active && (
                <button className="px-2 py-1.5 rounded-md text-xs border border-gray-200 text-gray-500 hover:bg-gray-50"
                    onClick={() => { clearSavedAlignment(); setCleared(true); }}
                    title="Forget the saved 1:1 alignment, so the next AR on-site (or AR conveyor) session aligns again">
                    {cleared ? 'Alignment reset' : 'Reset AR alignment'}
                </button>
            )}
            {active && <span className="text-xs text-teal-700 self-center">AR session running in the headset</span>}
            {error && <span className="text-xs text-red-600 self-center">AR: {error}</span>}
        </>
    );
}
