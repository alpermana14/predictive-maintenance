import React, { useState, useEffect, useMemo } from 'react';
import axios from 'axios';
import {
    LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer, 
    ReferenceLine, ScatterChart, Scatter, Cell, BarChart, Bar, ReferenceArea
} from 'recharts';

// ================= 1. CONFIGURATION =================
const API_URL = import.meta.env.VITE_API_URL || "http://localhost:8000/api";
const REFRESH_RATE = 5000; 

// ================= 2. ISO SEVERITY COMPONENT =================
const IsoSeverityChart = ({ zRms, xRms }) => {
    const maxVal = Math.max(zRms, xRms, 5.5); 
    const xAxisMax = Math.ceil(maxVal * 1.1);

    // UX FIX: Staggered Label Tooltips with White Backgrounds
    const CustomDot = (props) => {
        const { cx, cy, payload } = props;
        const isAbove = payload.dy < 0;
        const boxY = cy + payload.dy;

        return (
            <g style={{ zIndex: 10 }}>
                {/* The Connector Line */}
                <line x1={cx} y1={cy} x2={cx} y2={boxY} stroke="#9ca3af" strokeWidth={1} strokeDasharray="2 2" />
                
                {/* The White Label Box */}
                <rect 
                    x={cx - 40} 
                    y={isAbove ? boxY - 20 : boxY} 
                    width={80} 
                    height={20} 
                    fill="#ffffff" 
                    rx={4} 
                    stroke="#e5e7eb" 
                    style={{filter: 'drop-shadow(0px 2px 4px rgba(0,0,0,0.1))'}}
                />
                
                {/* The Text */}
                <text 
                    x={cx} 
                    y={isAbove ? boxY - 6 : boxY + 14} 
                    textAnchor="middle" 
                    fill="#1f2937" 
                    fontSize={10} 
                    fontWeight="bold"
                >
                    {payload.name}: {payload.x.toFixed(2)}
                </text>
                
                {/* The Unified Dot (different colors for X and Z) */}
                <circle 
                    cx={cx} 
                    cy={cy} 
                    r={7} 
                    fill={payload.color} 
                    stroke="#ffffff" 
                    strokeWidth={2} 
                    style={{filter: 'drop-shadow(0px 2px 2px rgba(0,0,0,0.3))'}} 
                />
            </g>
        );
    };

    return (
        <div className="bg-white p-6 rounded-xl shadow-sm border border-gray-100 mb-6">
            <h2 className="text-center text-lg font-semibold text-gray-700 mb-6">
                Vibration Severity according to ISO 10816-3
            </h2>
            <div className="h-[120px] w-full">
                <ResponsiveContainer width="100%" height="100%">
                    {/* Increased top/bottom margins to make room for the labels */}
                    <ScatterChart margin={{ top: 30, right: 20, bottom: 30, left: 20 }}>
                        <XAxis 
                            type="number" 
                            dataKey="x" 
                            domain={[0, xAxisMax]} 
                            name="RMS Velocity (mm/s)" 
                            tick={{fontSize: 12, fill: '#6b7280'}}
                            axisLine={false}
                            tickLine={false}
                        />
                        <YAxis type="number" dataKey="y" domain={[0, 1]} hide />
                        
                        {/* UX FIX: Zone Backgrounds */}
                        <ReferenceArea x1={0} x2={0.71} y1={0.3} y2={0.7} fill="#22c55e" fillOpacity={0.85} />
                        <ReferenceArea x1={0.71} x2={1.8} y1={0.3} y2={0.7} fill="#f59e0b" fillOpacity={0.85} />
                        <ReferenceArea x1={1.8} x2={4.5} y1={0.3} y2={0.7} fill="#f97316" fillOpacity={0.85} />
                        <ReferenceArea x1={4.5} x2={xAxisMax} y1={0.3} y2={0.7} fill="#dc2626" fillOpacity={0.85} />

                        {/* UX FIX: Vertical Demarcation Lines for exact thresholds */}
                        <ReferenceLine x={0.71} stroke="#ffffff" strokeWidth={2} strokeOpacity={0.5} y1={0.3} y2={0.7} />
                        <ReferenceLine x={1.8} stroke="#ffffff" strokeWidth={2} strokeOpacity={0.5} y1={0.3} y2={0.7} />
                        <ReferenceLine x={4.5} stroke="#ffffff" strokeWidth={2} strokeOpacity={0.5} y1={0.3} y2={0.7} />

                        {/* UX FIX: Both dots sit precisely at y: 0.5. Labels are staggered using 'dy' */}
                        <Scatter 
                            data={[
                                { x: zRms, y: 0.5, name: 'Z-RMS', dy: -25, color: '#1e40af' }, // Blue dot, label up
                                { x: xRms, y: 0.5, name: 'X-RMS', dy: 25, color: '#374151' }   // Gray dot, label down
                            ]} 
                            shape={<CustomDot />} 
                            isAnimationActive={true}
                        />
                    </ScatterChart>
                </ResponsiveContainer>
            </div>
            
            {/* Custom Legend */}
            <div className="flex justify-center items-center space-x-6 text-xs text-gray-600 mt-2">
                <div className="flex items-center"><span className="w-3 h-3 rounded-full bg-[#1e40af] border-2 border-white shadow mr-2"></span> Z-RMS</div>
                <div className="flex items-center"><span className="w-3 h-3 rounded-full bg-[#374151] border-2 border-white shadow mr-2"></span> X-RMS</div>
                <div className="flex items-center ml-4"><span className="w-4 h-4 bg-[#dc2626] mr-2"></span> Zone D</div>
                <div className="flex items-center"><span className="w-4 h-4 bg-[#f97316] mr-2"></span> Zone C</div>
                <div className="flex items-center"><span className="w-4 h-4 bg-[#f59e0b] mr-2"></span> Zone B</div>
                <div className="flex items-center"><span className="w-4 h-4 bg-[#22c55e] mr-2"></span> Zone A</div>
            </div>
        </div>
    );
};
// ================= 3. MAIN DASHBOARD =================
export default function Dashboard() {
    const [selectedTarget, setSelectedTarget] = useState("z_rms");
    const [summary, setSummary] = useState(null);
    const [forecastData, setForecastData] = useState([]); 
    const [anomalyData, setAnomalyData] = useState([]);   
    const [importanceData, setImportanceData] = useState([]); 
    const [threshold, setThreshold] = useState(0);
    const [workOrders, setWorkOrders] = useState([]);
    const [selectedWorkOrder, setSelectedWorkOrder] = useState(null);
    const [workOrdersLoading, setWorkOrdersLoading] = useState(false);

    useEffect(() => {
        const fetchSummary = () => {
            axios.get(`${API_URL}/summary`)
                .then(res => setSummary(res.data))
                .catch(err => console.error("Summary API Error:", err));
        };
        fetchSummary();
        const intervalId = setInterval(fetchSummary, REFRESH_RATE);
        return () => clearInterval(intervalId);
    }, []);

    useEffect(() => {
        const fetchCharts = async () => {
            try {
                const [forecastRes, anomalyRes, importanceRes] = await Promise.all([
                    axios.get(`${API_URL}/forecast/${selectedTarget}`),
                    axios.get(`${API_URL}/anomalies/${selectedTarget}`),
                    axios.get(`${API_URL}/importance`)
                ]);

                // --- LINE CHART (WITH ERROR FLAGS) ---
                const history = forecastRes.data.history_x.map((time, i) => {
                    const val = forecastRes.data.history_y[i];
                    const isError = forecastRes.data.history_flags ? forecastRes.data.history_flags[i] : false;
                    
                    return {
                        time: time.substring(11, 16),
                        history: val, 
                        history_error: isError ? val : null, 
                        forecast: null 
                    };
                });
                
                const forecast = forecastRes.data.forecast_x.map((time, i) => ({
                    time: time.substring(11, 16),
                    history: null,
                    history_error: null,
                    forecast: forecastRes.data.forecast_y[i]
                }));
                setForecastData([...history, ...forecast]);

                // --- SCATTER CHARTS ---
                const currentThreshold = anomalyRes.data.threshold;
                const timestamps = anomalyRes.data.timestamps || []; 
                const rawValues = anomalyRes.data.raw_values || []; 

                const anomalies = anomalyRes.data.scores.map((score, i) => ({
                    time: timestamps[i] ? new Date(timestamps[i]).getTime() : i, 
                    score: score,
                    rawValue: rawValues[i] || 0, 
                    status: score < currentThreshold ? "Anomaly" : "Normal",
                    dateStr: timestamps[i] 
                }));
                
                setAnomalyData(anomalies);
                setThreshold(currentThreshold);

                // --- IMPORTANCE ---
                const allImportance = importanceRes.data;
                if (allImportance && allImportance[selectedTarget]) {
                    const topFeatures = allImportance[selectedTarget]
                        .slice(0, 10)
                        .map(item => ({
                            feature: item.feature.replace(/_/g, ' ').toUpperCase(),
                            importance: item.importance
                        }));
                    setImportanceData(topFeatures);
                }

            } catch (err) {
                console.error("Chart API Error:", err);
            }
        };

        fetchCharts();
        const intervalId = setInterval(fetchCharts, REFRESH_RATE);
        return () => clearInterval(intervalId);

    }, [selectedTarget]);

    const isStale = useMemo(() => {
        if (!summary || !summary.data_timestamp) return false;
        const sensorTime = new Date(summary.data_timestamp);
        const diffMins = Math.floor((new Date() - sensorTime) / 60000);
        return diffMins > 30;
    }, [summary]);

    useEffect(() => {
        const fetchWorkOrders = async () => {
            try {
                setWorkOrdersLoading(true);
                const res = await axios.get(`${API_URL}/work_orders`);
                setWorkOrders(res.data.items || []);
            } catch (err) {
                console.error("Work Orders API Error:", err);
            } finally {
                setWorkOrdersLoading(false);
            }
        };

        fetchWorkOrders();
        const intervalId = setInterval(fetchWorkOrders, 60000);
        return () => clearInterval(intervalId);
    }, []);

    if (!summary) return <div className="flex h-screen items-center justify-center bg-gray-50 text-blue-600 animate-pulse text-xl font-mono">🚀 Connecting to AI System...</div>;

    // Determine the color of the status box based on ISO Zone
    const getStatusStyle = (zone) => {
        switch(zone) {
            case 'A': return 'bg-green-500 text-white shadow-green-200';
            case 'B': return 'bg-yellow-500 text-white shadow-yellow-200';
            case 'C': return 'bg-orange-500 text-white shadow-orange-200';
            case 'D': return 'bg-red-600 text-white shadow-red-200';
            default: return 'bg-gray-500 text-white shadow-gray-200';
        }
    };

    return (
        <div className="min-h-screen bg-gray-50 p-6 md:p-8 font-sans text-gray-800">
           {/* STALE WARNING */}
           {isStale && (
                <div className="mb-6 bg-red-50 border-l-4 border-red-500 p-4 rounded-r shadow-sm flex items-start gap-4 animate-bounce-slow">
                    <div className="text-3xl">⚠️</div>
                    <div>
                        <h3 className="font-bold text-red-800 text-lg">Connection Warning: Data is Not Real-Time</h3>
                        <p className="text-red-700 text-sm mt-1">
                            The latest reading was received at <span className="font-mono font-bold">{summary.data_timestamp}</span>.
                        </p>
                        <p className="text-red-600 text-xs mt-2 font-semibold uppercase tracking-wide">
                            Action Required: Check Controller Power • Check Internet Connection • Inspect Conveyor System
                        </p>
                    </div>
                </div>
            )}

            {/* HEADER */}
            <header className="mb-6 flex flex-col md:flex-row justify-between items-center bg-white p-6 rounded-xl shadow-sm border border-gray-100">
                <div>
                    <h1 className="text-2xl font-bold text-gray-900">Predictive Maintenance Dashboard</h1>
                    <p className="text-gray-500 text-sm mt-1">System: Conveyor 450W | Last Reading: {summary.data_timestamp}</p>
                </div>
                
                {/* UPDATED STATUS BOX (Matches ISO Colors) */}
                <div className={`px-8 py-3 rounded-lg shadow-lg flex flex-col items-center justify-center transition-colors ${getStatusStyle(summary.iso_zone)}`}>
                    <span className="font-bold tracking-wider uppercase text-xl flex items-center">
                        {summary.iso_zone !== 'A' && <span className="mr-2">⚠️</span>}
                        Zone {summary.iso_zone} - {summary.status}
                    </span>
                </div>
            </header>

            {/* NEW ISO SEVERITY CHART */}
            <IsoSeverityChart 
                zRms={summary.metrics.z_rms || 0} 
                xRms={summary.metrics.x_rms || 0} 
            />

            {/* SENSORS */}
            <div className="grid grid-cols-2 md:grid-cols-4 lg:grid-cols-7 gap-3 mb-6">
                {Object.entries(summary.metrics).map(([key, val]) => {
                    // Skip error flag columns in the metric boxes
                    if (key.includes("_error_flag")) return null;
                    
                    return (
                        <button key={key} onClick={() => setSelectedTarget(key)} className={`p-3 rounded-lg border text-left transition-all ${selectedTarget === key ? 'bg-blue-600 text-white shadow-lg scale-105' : 'bg-white hover:bg-blue-50'}`}>
                            <span className={`text-[10px] uppercase font-bold ${selectedTarget === key ? 'text-blue-100' : 'text-gray-400'}`}>{key}</span>
                            <div className="text-lg font-mono font-medium mt-1">{typeof val === 'number' ? val.toFixed(2) : val}</div>
                        </button>
                    );
                })}
            </div>

            {/* ROW 1: FORECAST & DIAGNOSTICS */}
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 mb-6">
                
                {/* 1. FORECAST */}
                <div className="bg-white p-6 rounded-xl shadow-sm border border-gray-100">
                    <h2 className="text-lg font-semibold mb-6 flex items-center"><span className="w-2 h-2 rounded-full bg-blue-500 mr-2"></span>Forecast: {selectedTarget}</h2>
                    <div className="h-[300px] w-full">
                        <ResponsiveContainer width="100%" height="100%">
                            <LineChart data={forecastData}>
                                <CartesianGrid strokeDasharray="3 3" stroke="#f0f0f0" />
                                <XAxis dataKey="time" tick={{fontSize: 12, fill: '#9ca3af'}} minTickGap={30} />
                                <YAxis 
                                    domain={['auto', 'auto']} 
                                    tick={{fontSize: 12, fill: '#9ca3af'}} 
                                    tickFormatter={(val) => val.toFixed(2)}
                                />
                                <Tooltip contentStyle={{borderRadius: '8px'}} />
                                <Legend />
                                
                                {/* 1. Normal History Line (Blue, Solid) */}
                                <Line type="monotone" dataKey="history" stroke="#2563eb" strokeWidth={2} dot={false} name="Real History" connectNulls />
                                
                                {/* 2. Interpolated Error Overlay (Red, Dotted with points) */}
                                <Line type="monotone" dataKey="history_error" stroke="#ef4444" strokeDasharray="5 5" strokeWidth={3} dot={{ r: 4, fill: '#ef4444' }} name="Controller Error" connectNulls={false} />
                                
                                {/* 3. AI Forecast Line (Purple) */}
                                <Line type="monotone" dataKey="forecast" stroke="#8b5cf6" strokeWidth={2} strokeDasharray="5 5" dot={false} name="AI Forecast" connectNulls />
                            </LineChart>
                        </ResponsiveContainer>
                    </div>
                </div>

                {/* 2. DIAGNOSTICS */}
                <div className="bg-white p-6 rounded-xl shadow-sm border border-gray-100">
                    <h2 className="text-lg font-semibold mb-2 flex items-center"><span className="w-2 h-2 rounded-full bg-gray-600 mr-2"></span>Diagnostics</h2>
                    <p className="text-sm text-gray-500 mb-4">Key influencing factors for {selectedTarget}</p>
                    <div className="h-[300px] w-full"> 
                        <ResponsiveContainer width="100%" height="100%">
                            <BarChart layout="vertical" data={importanceData} margin={{ top: 5, right: 30, left: 40, bottom: 5 }}>
                                <CartesianGrid strokeDasharray="3 3" horizontal={false} stroke="#f0f0f0" />
                                <XAxis type="number" tick={{fontSize: 12, fill: '#9ca3af'}} />
                                <YAxis dataKey="feature" type="category" width={120} tick={{fontSize: 11, fill: '#4b5563', fontWeight: 600}} />
                                <Tooltip cursor={{fill: '#f3f4f6'}} contentStyle={{borderRadius: '8px'}} />
                                <Bar dataKey="importance" fill="#4b5563" radius={[0, 4, 4, 0]} barSize={18} />
                            </BarChart>
                        </ResponsiveContainer>
                    </div>
                </div>
            </div>

            {/* ROW 2: ANOMALY CORRELATION + WORK ORDER HISTORY */}
            <div className="grid grid-cols-1 lg:grid-cols-3 gap-6 mt-6">
                {/* Anomaly Analysis (spans 2 cols on large screens) */}
                <div className="bg-white p-6 rounded-xl shadow-sm border border-gray-100 flex flex-col gap-4 lg:col-span-2">
                    <h2 className="text-lg font-semibold mb-2 flex items-center"><span className="w-2 h-2 rounded-full bg-purple-500 mr-2"></span>Anomaly Analysis</h2>
                    
                    {/* TOP: RAW DISTRIBUTION */}
                    <div className="h-[140px] w-full border-b border-gray-100 pb-2">
                        <div className="flex justify-between items-center mb-2">
                            <h2 className="text-sm font-semibold text-gray-700">1. Signal Distribution ({selectedTarget})</h2>
                            <span className="text-xs text-gray-400">Red dots indicate anomalies</span>
                        </div>
                        <ResponsiveContainer width="100%" height="100%">
                            <ScatterChart margin={{ top: 5, right: 10, bottom: 0, left: 0 }}>
                                <CartesianGrid strokeDasharray="3 3" stroke="#f0f0f0" />
                                <XAxis type="number" dataKey="time" hide domain={['dataMin', 'dataMax']} />
                                <YAxis 
                                    type="number" 
                                    dataKey="rawValue" 
                                    domain={['auto', 'auto']} 
                                    tick={{fontSize: 10}} 
                                    width={30} 
                                />
                                <Tooltip cursor={{ strokeDasharray: '3 3' }} contentStyle={{borderRadius: '8px'}} labelFormatter={() => ''} />
                                <Scatter name="Raw Signal" data={anomalyData} fill="#8884d8" shape="circle">
                                    {anomalyData.map((entry, index) => (
                                        <Cell key={`cell-raw-${index}`} fill={entry.score < threshold ? '#ef4444' : '#93c5fd'} />
                                    ))}
                                </Scatter>
                            </ScatterChart>
                        </ResponsiveContainer>
                    </div>
                    
                    {/* BOTTOM: IDK SCORE */}
                    <div className="h-[140px] w-full">
                        <div className="flex justify-between items-center mb-2">
                            <h2 className="text-sm font-semibold text-gray-700">2. Anomaly Confidence (IDK Score)</h2>
                            <span className="text-xs text-gray-400">Low Score = Anomaly</span>
                        </div>
                        <ResponsiveContainer width="100%" height="100%">
                            <ScatterChart margin={{ top: 5, right: 10, bottom: 0, left: 0 }}>
                                <CartesianGrid strokeDasharray="3 3" stroke="#f0f0f0" />
                                <XAxis 
                                    type="number" dataKey="time" 
                                    domain={['dataMin', 'dataMax']}
                                    tickFormatter={(unixTime) => new Date(unixTime).toLocaleDateString('en-GB', { day: '2-digit', month: 'short' })}
                                    tick={{fontSize: 10, fill: '#9ca3af'}}
                                />
                                <YAxis type="number" dataKey="score" domain={[0, 'auto']} tick={{fontSize: 10}} width={30} />
                                <Tooltip cursor={{ strokeDasharray: '3 3' }} contentStyle={{borderRadius: '8px'}} labelFormatter={(label) => new Date(label).toLocaleString()} />
                                <ReferenceLine y={threshold} stroke="#ef4444" strokeDasharray="3 3" />
                                <Scatter name="IDK Score" data={anomalyData} fill="#8884d8">
                                    {anomalyData.map((entry, index) => (
                                        <Cell key={`cell-idk-${index}`} fill={entry.score < threshold ? '#ef4444' : '#8b5cf6'} />
                                    ))}
                                </Scatter>
                            </ScatterChart>
                        </ResponsiveContainer>
                    </div>
                </div>

                {/* Work Order History */}
                <div className="bg-white p-6 rounded-xl shadow-sm border border-gray-100 flex flex-col">
                    <div className="flex items-center justify-between mb-3">
                        <h2 className="text-lg font-semibold flex items-center">
                            <span className="w-2 h-2 rounded-full bg-yellow-500 mr-2"></span>
                            Work Order History
                        </h2>
                        {workOrdersLoading && (
                            <span className="text-xs text-gray-400 animate-pulse">Refreshing...</span>
                        )}
                    </div>
                    {workOrders.length === 0 ? (
                        <p className="text-xs text-gray-400">
                            No work orders saved yet. Use the chatbot to draft and finalize a work order.
                        </p>
                    ) : (
                        <div className="flex-1 flex flex-col gap-3 overflow-hidden">
                            <div className="space-y-2 overflow-y-auto max-h-48 pr-1">
                                {workOrders.map((wo) => (
                                    <button
                                        key={wo.id}
                                        onClick={() => setSelectedWorkOrder(wo)}
                                        className={`w-full text-left p-2 rounded border text-xs mb-1 transition-colors ${
                                            selectedWorkOrder?.id === wo.id
                                                ? 'bg-blue-50 border-blue-300 text-blue-800'
                                                : 'bg-gray-50 hover:bg-gray-100 border-gray-200 text-gray-700'
                                        }`}
                                    >
                                        <div className="flex justify-between items-center mb-1">
                                            <span className="font-semibold truncate max-w-[70%]">
                                                {wo.id || 'Unnamed WO'}
                                            </span>
                                            <span className="text-[10px] text-gray-400">
                                                {wo.created_at
                                                    ? new Date(wo.created_at).toLocaleString()
                                                    : ''}
                                            </span>
                                        </div>
                                        <p className="text-[11px] line-clamp-2">
                                            {wo.preview}
                                        </p>
                                    </button>
                                ))}
                            </div>
                            {selectedWorkOrder && (
                                <div className="mt-3 border-t pt-2 text-xs text-gray-700 max-h-40 overflow-y-auto">
                                    <h3 className="font-semibold mb-1">
                                        Details: {selectedWorkOrder.id}
                                    </h3>
                                    <p className="whitespace-pre-wrap">
                                        {selectedWorkOrder.content || selectedWorkOrder.preview}
                                    </p>
                                </div>
                            )}
                        </div>
                    )}
                </div>
            </div>
        </div>
    );
}
