import React, { useState, useEffect, useRef } from 'react';
import axios from 'axios';

export default function Chatbot() {
    const [isOpen, setIsOpen] = useState(false);
    const [messages, setMessages] = useState([]);
    const [input, setInput] = useState("");
    const [draft, setDraft] = useState(""); // Stores the active WO draft text
    const [loading, setLoading] = useState(false);

    // Track session ID to maintain conversation context with the AI
    const [sessionId, setSessionId] = useState(
        () => "session-" + Math.random().toString(36).substr(2, 9)
    );
    const scrollRef = useRef(null);

    // --- 1. UTILITY FUNCTIONS ---

    const resetChat = () => {
        setMessages([]);
        setDraft("");
        setInput("");
        // Generate a new session ID to start fresh
        setSessionId("session-" + Math.random().toString(36).substr(2, 9));
    };

    const sendMessage = async () => {
        if (!input.trim()) return;
        
        const newMsg = { role: 'user', content: input };
        setMessages(prev => [...prev, newMsg]);
        setInput("");
        setLoading(true);

        try {
            const res = await axios.post('http://localhost:8000/api/chat', {
                message: newMsg.content,
                session_id: sessionId
            });

            const aiMsg = { role: 'ai', content: res.data.response };
            setMessages(prev => [...prev, aiMsg]);
            
            // If the AI created/updated a draft, update our state to show the yellow box
            if (res.data.draft) {
                setDraft(res.data.draft);
            }

        } catch (err) {
            console.error(err);
            setMessages(prev => [...prev, { role: 'ai', content: "⚠️ Error connecting to server." }]);
        } finally {
            setLoading(false);
        }
    };

    // --- 2. NEW: HANDLE APPROVAL ---
    const handleApprove = async () => {
        try {
            setLoading(true);
            // Call the new backend endpoint we discussed
            const res = await axios.post('http://localhost:8000/api/work_orders/approve', {
                session_id: sessionId
            });
            
            // On success:
            setDraft(""); // Hide the yellow box
            setMessages(prev => [...prev, { 
                role: 'ai', 
                content: `✅ Success! Work Order ${res.data.work_order_id} has been saved to the database.` 
            }]);

        } catch (err) {
            console.error(err);
            alert("Failed to approve work order. Please check the console.");
        } finally {
            setLoading(false);
        }
    };

    // Auto-scroll to bottom whenever messages change
    useEffect(() => {
        if (scrollRef.current) scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
    }, [messages]);

    return (
        <div className="fixed bottom-6 right-6 z-50 flex flex-col items-end pointer-events-none">
            
            {/* 3. WORK ORDER DRAFT PREVIEW (Updated UI) */}
            {isOpen && draft && (
                <div className="bg-yellow-50 border border-yellow-200 p-4 rounded-lg shadow-xl mb-4 w-80 pointer-events-auto flex flex-col gap-2 transition-all">
                    <div className="flex justify-between items-center">
                        <h3 className="text-xs font-bold text-yellow-800 uppercase">📝 Work Order Draft</h3>
                        <span className="text-[10px] text-yellow-600 font-semibold animate-pulse">Waiting for Approval</span>
                    </div>
                    
                    <textarea 
                        className="w-full text-xs p-2 border border-yellow-300 rounded bg-white h-32 resize-none focus:outline-none text-gray-700"
                        value={draft}
                        readOnly
                        placeholder="AI will generate draft here..."
                    />
                    
                    <div className="flex gap-2 mt-1">
                        <button 
                            onClick={handleApprove}
                            disabled={loading}
                            className="flex-1 bg-green-600 hover:bg-green-700 text-white text-xs font-bold py-2 rounded shadow-sm transition-colors disabled:opacity-50"
                        >
                            {loading ? "Saving..." : "✓ Approve & Save"}
                        </button>
                        <button 
                            onClick={() => setDraft("")} 
                            disabled={loading}
                            className="px-3 bg-gray-200 hover:bg-gray-300 text-gray-700 text-xs font-bold rounded transition-colors border border-gray-300"
                        >
                            ✕ Close
                        </button>
                    </div>
                </div>
            )}

            {/* 4. CHAT WINDOW */}
            {isOpen && (
                <div className="bg-white w-96 h-[500px] rounded-xl shadow-2xl border border-gray-200 flex flex-col pointer-events-auto overflow-hidden">
                    {/* Header */}
                    <div className="bg-blue-600 p-4 flex justify-between items-center shadow-md">
                        <div className="flex items-center gap-2">
                             <span className="text-2xl">🤖</span>
                             <div>
                                <h3 className="text-white font-bold leading-tight">Maintenance Copilot</h3>
                                <p className="text-blue-200 text-[10px]">AI-Powered Assistant</p>
                             </div>
                        </div>
                        <div className="flex items-center gap-2">
                            {/* New chat / clear history */}
                            <button
                                onClick={resetChat}
                                className="text-blue-100 hover:text-white text-xs border border-blue-400 hover:border-blue-200 px-2 py-1 rounded transition-all"
                            >
                                New Chat
                            </button>
                            {/* Close button */}
                            <button
                                onClick={() => {
                                    // Optional: Reset chat on close, or just hide
                                    setIsOpen(false);
                                }}
                                className="text-blue-100 hover:text-white ml-2 text-xl font-bold"
                            >
                                ×
                            </button>
                        </div>
                    </div>

                    {/* Messages Area */}
                    <div className="flex-1 p-4 overflow-y-auto bg-gray-50" ref={scrollRef}>
                        {messages.length === 0 && (
                            <div className="text-center text-gray-400 text-sm mt-10 px-6">
                                <p className="mb-2 text-3xl">👋</p>
                                <p className="font-semibold text-gray-500">How can I help?</p>
                                <p className="mt-2 text-xs">Try asking:</p>
                                <ul className="mt-2 space-y-2 text-xs text-blue-500">
                                    <li className="cursor-pointer hover:underline" onClick={() => setInput("What is the current status of the machine?")}>"What is the machine status?"</li>
                                    <li className="cursor-pointer hover:underline" onClick={() => setInput("Draft a work order for high vibration.")}>"Draft a work order."</li>
                                </ul>
                            </div>
                        )}
                        {messages.map((m, i) => (
                            <div key={i} className={`mb-3 flex ${m.role === 'user' ? 'justify-end' : 'justify-start'}`}>
                                <div className={`max-w-[85%] p-3 rounded-lg text-sm shadow-sm ${
                                    m.role === 'user' 
                                        ? 'bg-blue-600 text-white rounded-br-none' 
                                        : 'bg-white border border-gray-200 text-gray-800 rounded-bl-none'
                                }`}>
                                    {m.content}
                                </div>
                            </div>
                        ))}
                        {loading && (
                            <div className="flex items-center gap-2 text-xs text-gray-400 ml-2 mt-2">
                                <div className="w-2 h-2 bg-gray-400 rounded-full animate-bounce"></div>
                                <div className="w-2 h-2 bg-gray-400 rounded-full animate-bounce" style={{animationDelay: '0.1s'}}></div>
                                <div className="w-2 h-2 bg-gray-400 rounded-full animate-bounce" style={{animationDelay: '0.2s'}}></div>
                            </div>
                        )}
                    </div>

                    {/* Input Area */}
                    <div className="p-3 border-t bg-white flex gap-2">
                        <input 
                            className="flex-1 border border-gray-300 rounded-lg px-3 py-2 text-sm focus:outline-none focus:border-blue-500 focus:ring-1 focus:ring-blue-500 transition-all"
                            placeholder="Type a message..."
                            value={input}
                            onChange={(e) => setInput(e.target.value)}
                            onKeyPress={(e) => e.key === 'Enter' && sendMessage()}
                            disabled={loading}
                        />
                        <button 
                            onClick={sendMessage}
                            disabled={loading || !input.trim()}
                            className="bg-blue-600 text-white px-4 py-2 rounded-lg text-sm font-bold hover:bg-blue-700 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
                        >
                            ➤
                        </button>
                    </div>
                </div>
            )}

            {/* 5. TOGGLE BUTTON (Floating Action Button) */}
            {!isOpen && (
                <button 
                    onClick={() => setIsOpen(true)}
                    className="group bg-blue-600 hover:bg-blue-700 text-white p-4 rounded-full shadow-lg pointer-events-auto transition-all transform hover:scale-110 flex items-center justify-center relative"
                >
                    <span className="text-2xl">🤖</span>
                    {/* Optional: Notification dot if there's a draft waiting */}
                    {draft && (
                        <span className="absolute top-0 right-0 block h-3 w-3 rounded-full ring-2 ring-white bg-red-500"></span>
                    )}
                </button>
            )}
        </div>
    );
}