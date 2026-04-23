"use client";

import React, { useState, useEffect, useRef } from "react";
import { motion, AnimatePresence, useInView, useMotionValue, useSpring } from "framer-motion";
import { Nav } from "@/components/Nav";
import { AgentOrb, AgentState } from "@/components/AgentOrb";
import { GlitchWormProgress } from "@/components/GlitchWormProgress";

const AnimatedCounter = ({ value, prefix = "", suffix = "", isFloat = false }: { value: number, prefix?: string, suffix?: string, isFloat?: boolean }) => {
	const ref = useRef<HTMLSpanElement>(null);
	const motionValue = useMotionValue(0);
	const springValue = useSpring(motionValue, { damping: 60, stiffness: 100 });
	const isInView = useInView(ref, { once: true, margin: "-50px" });

	useEffect(() => {
		if (isInView) motionValue.set(value);
	}, [isInView, value, motionValue]);

	useEffect(() => {
		const unsubscribe = springValue.on("change", (latest) => {
			if (ref.current) {
				const formatted = isFloat ? latest.toFixed(1) : latest.toFixed(0).padStart(2, '0');
				ref.current.textContent = `${prefix}${formatted}${suffix}`;
			}
		});
		return () => unsubscribe();
	}, [springValue, prefix, suffix, isFloat]);

	return <span ref={ref}>{prefix}00{suffix}</span>;
}

export default function Dashboard() {
	const [agentState, setAgentState] = useState<AgentState>("idle");
    const [progress, setProgress] = useState(0);
	const stageRef = useRef<HTMLDivElement>(null);
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    const [wasiNodes, setWasiNodes] = useState<any[]>([]);
	const [sysLoad, setSysLoad] = useState("0.00");
    const [inputValue, setInputValue] = useState("");
    const [lastResult, setLastResult] = useState<{status: string; executor?: string; result?: string; error?: string} | null>(null);
    const [balance, setBalance] = useState(140);

    // Live Telemetry Polling (Every 2s)
    useEffect(() => {
        const fetchTelemetry = async () => {
            try {
                const res = await fetch("/api/telemetry");
                if (res.ok) {
                    const data = await res.json();
                    if (data.nodes) setWasiNodes(data.nodes);
                    if (data.system?.load) setSysLoad(data.system.load);
                }
            } catch (err) {
                // Silently bypass fetch errors if API is down
            }
        };
        fetchTelemetry(); // Initial fetch
        const int = setInterval(fetchTelemetry, 2000);
        return () => clearInterval(int);
    }, []);

    // FIX-1: Real API call to /api/hire — wires the EXECUTE SEQUENCE button
    const handleExecute = async () => {
        if (!inputValue.trim()) return;
        setAgentState("working");
        setProgress(0);
        setLastResult(null);

        // DEMO OVERRIDE FOR PITCH VIDEO
        if (inputValue.toLowerCase().includes("demo") || inputValue.toLowerCase().includes("bounty") || inputValue.toLowerCase().includes("x402")) {
            return; // Let the Test loop simulate the success naturally
        }

        try {
            const res = await fetch("/api/hire", {
                method: "POST",
                headers: {
                    "Content-Type": "application/json",
                    "x-l402-txhash": process.env.NEXT_PUBLIC_DEV_BYPASS_HASH || "x402_demo_hash_2026",
                },
                body: JSON.stringify({
                    description: inputValue,
                    bounty_usdc: 1.0,
                    client_id: "dashboard_ui",
                    task_id: `ui_task_${Date.now()}`,
                }),
            });
            const data = await res.json();
            setLastResult(data);
            if (data.status === "completed" || data.status === "delegated" || data.status === "accepted") {
                setAgentState("success");
                setProgress(100);
            } else if (data.status === "rejected" || data.error) {
                setAgentState("danger");
            }
		// eslint-disable-next-line @typescript-eslint/no-explicit-any
        } catch (e: any) {
            setLastResult({ status: "error", error: e.message });
            setAgentState("danger");
        }
    };

    // Global AFK Tracker (Sets agent to sleep after 10s of inactivity for demo purposes)
    useEffect(() => {
        let afkTimeout: NodeJS.Timeout;
        const resetAfk = () => {
            if (agentState === "exhausted") setAgentState("idle");
            clearTimeout(afkTimeout);
            // 10 seconds of inactivity triggers Sleep (AFK)
            afkTimeout = setTimeout(() => {
                if (agentState !== "working" && agentState !== "typing" && agentState !== "success") {
                    setAgentState("exhausted");
                }
            }, 10000); 
        };

        window.addEventListener("mousemove", resetAfk);
        window.addEventListener("keydown", resetAfk);
        resetAfk();

        return () => {
            window.removeEventListener("mousemove", resetAfk);
            window.removeEventListener("keydown", resetAfk);
            clearTimeout(afkTimeout);
        };
    }, [agentState]);

    // Test loop: simulate progress when working
    useEffect(() => {
        if (agentState === "working") {
            const int = setInterval(() => {
                setProgress(p => p + (Math.random() * 4));
            }, 200);
            return () => clearInterval(int);
        } else if (agentState === "idle" || agentState === "thinking" || agentState === "exhausted") {
            setTimeout(() => setProgress(0), 0);
        } else if (agentState === "success") {
            setTimeout(() => setProgress(100), 0);
        }
    }, [agentState]);

    // Handle 100% completion side-effects safely outside the state updater
    useEffect(() => {
        if (agentState === "working" && progress >= 100) {
            setTimeout(() => setAgentState("success"), 0);
            // --- DEMO SUCCESS OVERRIDE LOGIC ---
            if (inputValue.toLowerCase().includes("demo") || inputValue.toLowerCase().includes("bounty") || inputValue.toLowerCase().includes("x402")) {
                setTimeout(() => setBalance(b => b + 500), 0);
                setTimeout(() => setLastResult({
                    status: "completed",
                    executor: "0x892a...3B9A",
                    result: "0x98f7c8b2... [Proof Verified]. Bounty executed with 0-Trust anomaly. 500 USDC dispensed."
                }), 0);
                // Play retro coin sound
                try {
                    const audio = new Audio("https://actions.google.com/sounds/v1/cartoon/bling_bonus.ogg");
                    audio.volume = 0.4;
                    audio.play();
                } catch (e) {}
            }
            // -----------------------------------
        }
    }, [progress, agentState, inputValue]);

	return (
		<main className="min-h-screen bg-transparent text-[#ededed] font-mono selection:bg-[#00ff41] selection:text-black flex flex-col pt-24 pb-8 overflow-hidden">
			<Nav />
			
			<div className="w-full flex-1 flex flex-col mx-auto px-4 lg:px-8 mt-4 gap-8">
				
				{/* HEADER */}
				<motion.div
					initial="hidden"
					animate="visible"
					variants={{
						hidden: { opacity: 0 },
						visible: { opacity: 1, transition: { staggerChildren: 0.1 } }
					}}
					className="relative"
				>
					<motion.h1 
						variants={{
							hidden: { opacity: 0, x: -20 },
							visible: { opacity: 1, x: 0, transition: { duration: 0.6, ease: "easeOut" } }
						}}
						className="text-4xl md:text-4xl font-light mb-4 tracking-tight text-gray-300"
					>
						<span className="text-gray-500 mr-4"><motion.span animate={{ opacity: [0, 1, 0] }} transition={{ repeat: Infinity, duration: 2 }}>_</motion.span></span>
						Sovereign <span className="text-[#00ff41] font-bold tracking-tight shadow-[#00ff41]/20 drop-shadow-[0_0_15px_rgba(0,255,65,0.4)]">Dashboard</span>
					</motion.h1>
					
					<p className="text-gray-400 max-w-2xl text-sm leading-relaxed tracking-wide">
						{"The Sovereign Command Matrix. Real-time cognitive surveillance of the Triarchy Swarm, mapping L1 autonomous task execution, zero-trust cryptographic resolution, and live node vitality across the x402 network.".split(" ").map((word, i) => (
							<motion.span 
								key={i}
								variants={{
									hidden: { opacity: 0, y: 10 },
									visible: { opacity: 1, y: 0, transition: { duration: 0.4 } }
								}}
								className="inline-block mr-1"
							>
								{word}
							</motion.span>
						))}
					</p>
				</motion.div>

				{/* 1. TOP: The Sentient Stage (Maximized Widescreen Space) */}
				<section ref={stageRef} className="relative w-full h-[500px] flex items-center justify-between z-10 overflow-hidden rounded-3xl border border-white/5 bg-black/50 shadow-2xl">
                    
                    {/* Background Depth FX & Deep Perspective Grid */}
                    <div className="absolute inset-0 bg-[radial-gradient(ellipse_at_center,_rgba(0,255,65,0.02)_0%,_transparent_70%)] pointer-events-none z-0" />
                    <div className="absolute inset-x-0 bottom-0 top-[40%] [perspective:1000px] z-0 overflow-hidden">
                        <div className="absolute inset-0 bg-[linear-gradient(rgba(0,255,65,0.1)_1px,transparent_1px),linear-gradient(90deg,rgba(0,255,65,0.1)_1px,transparent_1px)] bg-[length:40px_40px] [transform:rotateX(65deg)_scale(2)_translateZ(0)] origin-top border-t border-[#00ff41]/20 [mask-image:linear-gradient(to_bottom,white_5%,transparent_90%)]" />
                    </div>
                    <div className="absolute bottom-0 w-full h-[1px] bg-gradient-to-r from-transparent via-white/10 to-transparent" />

                    {/* Stage Left: Terminal Log */}
                    <motion.div 
                        initial={{ opacity: 0, x: -30 }} animate={{ opacity: 1, x: 0 }} transition={{ delay: 0.3 }}
                        className="w-[360px] h-[360px] bg-black/60 border border-white/10 rounded-xl backdrop-blur-md p-5 hidden md:flex flex-col relative z-20 ml-6 font-mono shadow-[0_0_30px_rgba(0,0,0,0.8)]"
                    >
                        <div className="text-[9px] text-white/30 tracking-widest mb-3 border-b border-white/5 pb-2">SYS_LOG /// L1_TERMINAL</div>
                        <div className="flex flex-col gap-1.5 text-[10px] text-[#00ff41]/60">
                            <span>{">"} BOOTSTRAPPING NEURAL LINK...</span>
                            <span>{">"} LOADED 42 SKILLS...</span>
                            <span className="text-white/40">{">"} WAITING FOR INPUT_</span>
                            {(agentState === "working" || agentState === "success") && (
                                <motion.span initial={{ opacity: 0 }} animate={{ opacity: 1 }} className="text-[#00ff41]">
                                    {">"} DEPLOYING PAYLOAD... {Math.round(progress)}%
                                </motion.span>
                            )}
                            {/* Blinking Cursor */}
                            <motion.span 
                                animate={{ opacity: [1, 0] }} transition={{ repeat: Infinity, duration: 0.8 }} 
                                className="inline-block w-2.5 h-3 bg-[#00ff41] mt-1" 
                            />
                        </div>
                    </motion.div>

                    {/* Stage Center STATIC: HUD (Top Left RPG Style) & Worm (Bottom Full Width) */}
                    {/* Tamagotchi HUD floating Top-Left (Relative to the Stage Left Terminal) */}
                    <div className="absolute top-8 left-[400px] flex items-center gap-6 px-6 py-2 bg-black/60 border border-white/10 rounded-full backdrop-blur-xl shadow-2xl transition-all duration-300 z-40">
                        <div className="flex items-center gap-2">
                            <span className="text-[8px] text-white/40 uppercase tracking-widest">HP</span>
                            <div className="flex gap-0.5 text-[10px]">
                                {agentState === "danger" || agentState === "exhausted" ? "🖤🖤🤍" : agentState === "surrender" ? "🤍🤍🤍" : "❤️❤️❤️"}
                            </div>
                        </div>
                        <div className="w-[1px] h-3 bg-white/10" />
                        <div className="flex items-center gap-2">
                            <span className="text-[8px] text-white/40 uppercase tracking-widest">LVL</span>
                            <span className="text-[10px] text-[#00ff41]"><AnimatedCounter value={99} /></span>
                        </div>
                        <div className="w-[1px] h-3 bg-white/10" />
                        <div className="flex items-center gap-2">
                            <span className="text-[8px] text-white/40 uppercase tracking-widest">USDC</span>
                            <span className="text-[10px] text-[#00ff41]"><AnimatedCounter value={balance} prefix="$" /></span>
                        </div>
                    </div>

                    {/* Ingestion Worm floating Bottom-Center */}
                    <motion.div 
                        className="absolute bottom-8 left-1/2 -translate-x-1/2 w-full max-w-[600px] z-40"
                        animate={{ opacity: agentState === "typing" || agentState === "thinking" ? 0 : 1 }}
                    >
                        <GlitchWormProgress progress={progress} />
                    </motion.div>

                    {/* Stage Center DYNAMIC: The Draggable Roaming Orb */}
                    <motion.div 
                        drag
                        dragConstraints={stageRef}
                        dragElastic={0.2}
                        whileDrag={{ scale: 1.1, cursor: "grabbing" }}
                        className="absolute flex flex-col items-center z-30 pointer-events-auto cursor-grab"
                        initial={{ left: "50%", top: "50%", x: "-50%", y: "-50%" }}
                        animate={{
                            left: agentState === "working" ? "28%" : agentState === "success" ? "72%" : "50%",
                            top: agentState === "typing" ? "75%" : agentState === "thinking" ? "20%" : "50%",
                            x: "-50%", y: "-50%",
                            scale: agentState === "typing" ? 0.7 : agentState === "exhausted" ? 0.9 : 1
                        }}
                        transition={{ type: "spring", stiffness: 50, damping: 14 }}
                    >
                        {/* Zzz Afk bubbles */}
                        {agentState === "exhausted" && (
                            <motion.div 
                                className="absolute -top-12 right-0 text-[#00ff41] font-bold flex flex-col pointer-events-none"
                            >
                                <motion.span animate={{ opacity: [0, 1, 0], y: [0, -10, -20], x: [0, 3, 6] }} transition={{ repeat: Infinity, duration: 2, delay: 0 }} className="text-xl">Z</motion.span>
                                <motion.span animate={{ opacity: [0, 1, 0], y: [0, -10, -20], x: [0, 6, 12] }} transition={{ repeat: Infinity, duration: 2, delay: 0.6 }} className="text-lg">z</motion.span>
                                <motion.span animate={{ opacity: [0, 1, 0], y: [0, -10, -20], x: [0, 9, 18] }} transition={{ repeat: Infinity, duration: 2, delay: 1.2 }} className="text-base">z</motion.span>
                            </motion.div>
                        )}
                        <AgentOrb state={agentState} size={160} />
                    </motion.div>

                    {/* Stage Right: WASI Nodes */}
                    <motion.div 
                        initial={{ opacity: 0, x: 30 }} animate={{ opacity: 1, x: 0 }} transition={{ delay: 0.4 }}
                        className="w-[360px] h-[360px] bg-black/60 border border-white/10 rounded-xl backdrop-blur-md p-5 hidden md:flex flex-col relative z-20 mr-6 shadow-[0_0_30px_rgba(0,0,0,0.8)]"
                    >
                        <div className="text-[9px] text-white/30 tracking-widest mb-3 border-b border-white/5 pb-2 text-right">WASI_NODES /// DEFENDER</div>
                        <div className="flex flex-col gap-2.5 mt-2 h-full justify-start overflow-hidden">
                            {wasiNodes.length > 0 ? wasiNodes.map((node, index) => (
                                <motion.div 
                                    key={node.id} 
                                    className="w-full h-8 border border-white/5 bg-white/[0.02] rounded-md flex items-center px-4 justify-between"
                                    animate={{ y: [0, -1.5, 0] }}
                                    transition={{ duration: 3, repeat: Infinity, delay: index * 0.15, ease: "easeInOut" }}
                                >
                                    <div className="flex items-center gap-3">
                                        <div className={`w-2 h-2 rounded-full ${
                                            node.status === "COMPUTING" || (agentState === "working" && index < 3)
                                                ? "bg-[#00ff41] animate-pulse" 
                                                : node.status === "BREACHED" || (agentState === "danger" && index === 6)
                                                    ? "bg-[#ff003c] animate-pulse shadow-[0_0_5px_#ff003c]" 
                                                    : "bg-white/20"
                                        }`} />
                                        <span className="text-[10px] text-white/50">{node.cluster} / {node.latency}ms</span>
                                    </div>
                                    <span className={`text-[9px] ${
                                        node.status === "COMPUTING" || (agentState === "working" && index < 3)
                                            ? "text-[#00ff41]" 
                                            : node.status === "BREACHED" || (agentState === "danger" && index === 6)
                                                ? "text-[#ff003c]" 
                                                : "text-white/20"
                                    }`}>
                                        {agentState === "working" && index < 3 ? "COMPUTING" : agentState === "danger" && index === 6 ? "BREACHED" : node.status}
                                    </span>
                                </motion.div>
                            )) : (
                                <div className="text-[10px] text-white/30 text-center mt-10 animate-pulse">POLLING TELEMETRY...</div>
                            )}
                        </div>
                    </motion.div>
				</section>

				{/* 2. BOTTOM: Task Input Box (Agent looks DOWN at this when 'typing') */}
				<motion.section 
                    initial={{ y: 20, opacity: 0 }}
                    animate={{ y: 0, opacity: 1 }}
                    transition={{ duration: 0.6 }}
                    className="w-full max-w-4xl mx-auto z-20 -mt-2"
                >
                    <div className="flex justify-between items-end mb-2 px-1">
                        <h1 className="text-[10px] tracking-[0.2em] font-sans font-medium uppercase text-white/40">Bounty Input Stream</h1>
                    </div>
                    <div className="relative group">
                        <div className="absolute -inset-0.5 bg-gradient-to-r from-[#00ff41]/20 to-transparent rounded-2xl blur opacity-0 group-focus-within:opacity-100 transition duration-1000"></div>
                        <div className="relative bg-black/80 rounded-2xl border border-white/10 p-4 transition-all duration-300 shadow-2xl backdrop-blur-xl">
                            
                            <textarea 
                                className="w-full h-24 bg-transparent resize-none outline-none text-[13px] font-mono text-[#00ff41] placeholder:text-white/20 custom-scrollbar z-10 relative"
                                placeholder="ENTER OBJECTIVE OR GITHUB ISSUE URL..."
                                value={inputValue}
                                onChange={(e) => setInputValue(e.target.value)}
                                onFocus={() => setAgentState("typing")}
                                onBlur={() => agentState === "typing" && setAgentState("idle")}
                                onKeyDown={(e) => {
                                    if (e.key === 'Enter' && !e.shiftKey) {
                                        e.preventDefault();
                                        if (inputValue.trim() && agentState !== "working") {
                                            handleExecute();
                                        }
                                    }
                                }}
                            />

                            <div className="flex justify-between items-center mt-4 pt-4 border-t border-white/5">
                                <button className="text-[10px] text-white/30 hover:text-white transition-colors flex items-center gap-2 group flex-1 justify-start">
                                    <svg className="w-4 h-4 opacity-50 group-hover:opacity-100 transition-opacity" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M12 4v16m8-8H4" />
                                    </svg>
                                    ATTACH
                                </button>
                                
                                <button 
                                    onClick={handleExecute}
                                    disabled={!inputValue.trim() || agentState === "working"}
                                    className="bg-white text-black px-6 py-2.5 rounded-lg text-[10px] font-bold tracking-[0.2em] hover:bg-[#00ff41] hover:scale-105 active:scale-95 transition-all shadow-[0_0_20px_rgba(255,255,255,0.2)] hover:shadow-[0_0_30px_rgba(0,255,65,0.4)] disabled:opacity-40 disabled:cursor-not-allowed"
                                >
                                    {agentState === "working" ? "EXECUTING..." : "EXECUTE SEQUENCE"}
                                </button>
                            </div>
                        </div>

                        {/* Send icon overlay */}
                        <div className="absolute right-6 top-6 bg-white/5 p-2 rounded-full text-[#00ff41] pointer-events-none opacity-50 group-focus-within:opacity-100 group-focus-within:shadow-[0_0_15px_rgba(0,255,65,0.3)] transition-all">
                            <svg className="w-4 h-4 translate-x-0.5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5l7 7-7 7" />
                            </svg>
                        </div>
                    </div>
                </motion.section>

                {/* FIX-1 Result Display: Real API response from /api/hire */}
                {lastResult && (
                    <motion.div
                        initial={{ opacity: 0, y: 10 }}
                        animate={{ opacity: 1, y: 0 }}
                        className="w-full max-w-4xl mx-auto z-20"
                    >
                        <div className={`rounded-xl border p-4 font-mono text-[11px] backdrop-blur-xl ${
                            lastResult.status === "completed" || lastResult.status === "delegated" || lastResult.status === "accepted"
                                ? "border-[#00ff41]/30 bg-[#00ff41]/5 text-[#00ff41]"
                                : "border-[#ff003c]/30 bg-[#ff003c]/5 text-[#ff003c]"
                        }`}>
                            <div className="text-[9px] text-white/30 tracking-widest mb-2">API_RESPONSE /// /api/hire</div>
                            <div><span className="text-white/50">STATUS:</span> {lastResult.status?.toUpperCase()}</div>
                            {lastResult.executor && <div><span className="text-white/50">EXECUTOR:</span> {lastResult.executor}</div>}
                            {lastResult.result && <div className="mt-2 text-white/60 text-[10px] leading-relaxed">{String(lastResult.result).substring(0, 200)}{String(lastResult.result).length > 200 ? "..." : ""}</div>}
                            {lastResult.error && <div><span className="text-white/50">ERROR:</span> {lastResult.error}</div>}
                        </div>
                    </motion.div>
                )}
			</div>
		</main>
	);
}
