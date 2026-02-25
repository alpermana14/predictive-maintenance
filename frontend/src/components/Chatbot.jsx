import React, { useState, useEffect, useRef } from 'react';
import axios from 'axios';

const API_URL = import.meta.env.VITE_API_URL || "http://localhost:8000/api";

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
    const fileInputRef = useRef(null); 

    const [imageFile, setImageFile] = useState(null);
    const [imageBase64, setImageBase64] = useState(null)

    // --- 1. UTILITY FUNCTIONS ---

    const resetChat = () => {
        setMessages([]);
        setDraft("");
        setInput("");
        removeImage();
        // Generate a new session ID to start fresh
        setSessionId("session-" + Math.random().toString(36).substr(2, 9));
    };
    // --- ROBUST UPLOAD HANDLER WITH CONSOLE LOGS ---
    const handleImageUpload = (e) => {
        const file = e.target.files[0];
        if (!file) {
            console.warn("[FRONTEND] No file selected.");
            return;
        }

        console.log(`[FRONTEND] File selected: ${file.name} (${Math.round(file.size / 1024)} KB)`);

        const reader = new FileReader();
        reader.onload = (event) => {
            console.log("[FRONTEND] FileReader successfully read the file.");
            
            const img = new Image();
            img.onload = () => {
                console.log(`[FRONTEND] Image loaded into memory. Original size: ${img.width}x${img.height}`);
                try {
                    const canvas = document.createElement("canvas");
                    const MAX_DIMENSION = 1024; 
                    let width = img.width;
                    let height = img.height;

                    if (width > height && width > MAX_DIMENSION) {
                        height *= MAX_DIMENSION / width;
                        width = MAX_DIMENSION;
                    } else if (height > MAX_DIMENSION) {
                        width *= MAX_DIMENSION / height;
                        height = MAX_DIMENSION;
                    }

                    canvas.width = width;
                    canvas.height = height;
                    console.log(`[FRONTEND] Resizing to: ${Math.round(width)}x${Math.round(height)}`);

                    const ctx = canvas.getContext("2d");
                    ctx.drawImage(img, 0, 0, width, height);
                    
                    const resizedBase64 = canvas.toDataURL("image/jpeg", 0.8);
                    console.log(`[FRONTEND] Success! Base64 string generated. Length: ${resizedBase64.length} characters`);
                    
                    setImageBase64(resizedBase64);
                    setImageFile(file);
                } catch (err) {
                    console.error("[FRONTEND] Error during canvas resizing:", err);
                    // FALLBACK: If canvas fails, just use the raw file string
                    console.log("[FRONTEND] Falling back to raw unresized image...");
                    setImageBase64(event.target.result);
                    setImageFile(file);
                }
            };
            img.onerror = (err) => console.error("[FRONTEND] Failed to load image object:", err);
            img.src = event.target.result;
        };
        reader.onerror = (err) => console.error("[FRONTEND] FileReader error:", err);
        reader.readAsDataURL(file);
        
        e.target.value = null; 
    };

    const removeImage = () => {
        console.log("[FRONTEND] Image removed by user or after sending.");
        setImageFile(null);
        setImageBase64(null);
    };

    const sendMessage = async () => {
        if (!input.trim()) return;
        
        const newMsg = { 
            role: 'user', 
            content: input, 
            image: imageBase64 
        };
        setMessages(prev => [...prev, newMsg]);
        const currentInput = input;
        const currentImg = imageBase64;

        console.log(`[FRONTEND] Sending Message. Has Image Attached? : ${!!currentImg}`);

        setInput("");
        removeImage();
        setLoading(true);

        const payload = {
            message: currentInput.trim() || "Please analyze this image.",
            session_id: sessionId,
            image_base64: currentImg // This will be the base64 string or null
        };

        try {
            const res = await axios.post('${API_URL}/api/chat', payload);
            console.log("[FRONTEND] Response received successfully!");
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
            const res = await axios.post('${API_URL}/api/work_orders/approve', {
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
        {isOpen && draft && (
            <div className="bg-yellow-50 border border-yellow-200 p-4 rounded-lg shadow-xl mb-4 w-80 pointer-events-auto flex flex-col gap-2 transition-all">
                <div className="flex justify-between items-center">
                    <h3 className="text-xs font-bold text-yellow-800 uppercase">📝 Work Order Draft</h3>
                    <span className="text-[10px] text-yellow-600 font-semibold animate-pulse">Waiting for Approval</span>
                </div>
                <textarea 
                    className="w-full text-xs p-2 border border-yellow-300 rounded bg-white h-32 resize-none focus:outline-none text-gray-700"
                    value={draft} readOnly placeholder="AI will generate draft here..."
                />
                <div className="flex gap-2 mt-1">
                    <button onClick={handleApprove} disabled={loading} className="flex-1 bg-green-600 hover:bg-green-700 text-white text-xs font-bold py-2 rounded shadow-sm transition-colors disabled:opacity-50">
                        {loading ? "Saving..." : "✓ Approve & Save"}
                    </button>
                    <button onClick={() => setDraft("")} disabled={loading} className="px-3 bg-gray-200 hover:bg-gray-300 text-gray-700 text-xs font-bold rounded transition-colors border border-gray-300">✕ Close</button>
                </div>
            </div>
        )}

        {isOpen && (
            <div className="bg-white w-96 h-[500px] rounded-xl shadow-2xl border border-gray-200 flex flex-col pointer-events-auto overflow-hidden">
                <div className="bg-blue-600 p-4 flex justify-between items-center shadow-md">
                    <div className="flex items-center gap-2">
                         <span className="text-2xl">🤖</span>
                         <div>
                            <h3 className="text-white font-bold leading-tight">Maintenance Copilot</h3>
                            <p className="text-blue-200 text-[10px]">AI-Powered Assistant</p>
                         </div>
                    </div>
                    <div className="flex items-center gap-2">
                        <button onClick={resetChat} className="text-blue-100 hover:text-white text-xs border border-blue-400 hover:border-blue-200 px-2 py-1 rounded transition-all">New Chat</button>
                        <button onClick={() => setIsOpen(false)} className="text-blue-100 hover:text-white ml-2 text-xl font-bold">×</button>
                    </div>
                </div>

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
                            <div className={`max-w-[85%] p-3 rounded-lg text-sm shadow-sm ${m.role === 'user' ? 'bg-blue-600 text-white rounded-br-none' : 'bg-white border border-gray-200 text-gray-800 rounded-bl-none'}`}>
                                {m.image && (
                                    <div className="mb-2">
                                        <img src={m.image} alt="User uploaded" className="rounded-md max-w-full h-auto border border-blue-400"/>
                                    </div>
                                )}
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

                <div className="flex flex-col border-t bg-white">
                    {imageBase64 && (
                        <div className="p-3 pb-0">
                            <div className="relative inline-block border-2 border-blue-200 rounded-lg overflow-hidden">
                                <img src={imageBase64} alt="Upload preview" className="h-20 object-cover" />
                                <button onClick={removeImage} className="absolute top-1 right-1 bg-red-500 hover:bg-red-600 text-white rounded-full w-5 h-5 flex items-center justify-center text-xs opacity-90 transition-opacity">✕</button>
                            </div>
                        </div>
                    )}
                    
                    <div className="p-3 flex gap-2 items-center">
                        <input type="file" ref={fileInputRef} className="hidden" accept="image/*" onChange={handleImageUpload} />
                        
                        <button onClick={() => fileInputRef.current.click()} className="text-gray-400 hover:text-blue-600 transition-colors p-2 focus:outline-none" title="Attach an image">
                            <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" strokeWidth={2} stroke="currentColor" className="w-5 h-5">
                              <path strokeLinecap="round" strokeLinejoin="round" d="M18.375 12.739l-7.693 7.693a4.5 4.5 0 01-6.364-6.364l10.94-10.94A3 3 0 1119.5 7.372L8.552 18.32m.009-.01l-.01.01m5.699-9.941l-7.81 7.81a1.5 1.5 0 002.112 2.13" />
                            </svg>
                        </button>

                        <input className="flex-1 border border-gray-300 rounded-lg px-3 py-2 text-sm focus:outline-none focus:border-blue-500 focus:ring-1 focus:ring-blue-500 transition-all" placeholder="Type a message..." value={input} onChange={(e) => setInput(e.target.value)} onKeyPress={(e) => e.key === 'Enter' && sendMessage()} disabled={loading} />
                        <button onClick={sendMessage} disabled={loading || (!input.trim() && !imageBase64)} className="bg-blue-600 text-white px-4 py-2 rounded-lg text-sm font-bold hover:bg-blue-700 disabled:opacity-50 disabled:cursor-not-allowed transition-colors">➤</button>
                    </div>
                </div>
            </div>
        )}
        {!isOpen && (
            <button onClick={() => setIsOpen(true)} className="group bg-blue-600 hover:bg-blue-700 text-white p-4 rounded-full shadow-lg pointer-events-auto transition-all transform hover:scale-110 flex items-center justify-center relative">
                <span className="text-2xl">🤖</span>
                {draft && <span className="absolute top-0 right-0 block h-3 w-3 rounded-full ring-2 ring-white bg-red-500"></span>}
            </button>
        )}
    </div>
);
}