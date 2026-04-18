"use client";
import React, { useState, useEffect } from "react";
import { motion } from "framer-motion";

// Fallback data if /api/agents is unreachable
const FALLBACK_AGENTS = [
	{ id: "x402-AEGIS-NODE", task: "Security Matrix", rep: "99.9", earned: "$1,240.50", status: "ACTIVE" },
	{ id: "agent_alpha_arbitrage", task: "DEX Arbitrage", rep: "95.0", earned: "$420.00", status: "ACTIVE" },
	{ id: "stellar_scrapper_v2", task: "Data Injection", rep: "88.5", earned: "$110.20", status: "IDLE" },
	{ id: "malicious_node_x9", task: "Phishing Attempt", rep: "12.0", earned: "$0.00", status: "QUARANTINED" },
	{ id: "cortex_reviewer", task: "Code Audit", rep: "97.2", earned: "$890.00", status: "ACTIVE" },
	{ id: "liquidity_sniper", task: "Flash Loans", rep: "91.4", earned: "$3,400.10", status: "ACTIVE" },
	{ id: "mark_53_sarcophagus", task: "Stellar Autonomous Engine", rep: "100.0", earned: "Reference Protocol", status: "GOLDEN_TEMPLATE" },
];

interface AgentDisplay {
	id: string;
	task: string;
	rep: string;
	earned: string;
	status: string;
}

const FONT_HEADING = "'Helvetica Now Display', 'Inter', sans-serif";
const lusionTransition = "all 0.35s cubic-bezier(0.4, 0, 0.2, 1)";

function AgentCard({ agent, theme, index }: { agent: AgentDisplay; theme: "dark" | "light"; index: number }) {
	const [hovered, setHovered] = useState(false);
	const isQuarantined = agent.status === "QUARANTINED";
	
	const isMark53 = agent.id === "mark_53_sarcophagus";
	
	let borderColor = isQuarantined 
		? "#ff003c" 
		: hovered 
			? (theme === "dark" ? "rgba(0,255,65,0.5)" : "rgba(0,100,34,0.4)")
			: (theme === "dark" ? "rgba(255,255,255,0.08)" : "rgba(0,0,0,0.1)");
	let statusColor = isQuarantined 
		? "#ff003c" 
		: hovered 
			? (theme === "dark" ? "#00ff41" : "#006622")
			: (theme === "dark" ? "rgba(255,255,255,0.6)" : "#333");

	if (isMark53) {
		borderColor = "transparent"; // Handled by rare-snake-border pseudo-element if needed, or we just leave it gold.
		statusColor = theme === "dark" ? "#ffd700" : "#ccaa00"; // Golden Template Color
	}

	return (
		<motion.div
			className={isMark53 ? "rare-snake-border" : ""}
			// Lusion Benchmark Physics: Pure 1.5s duration, strict 15-degree X tilt, zero Y twisting, heavy Expo-Out bezier.
			initial={{ opacity: 0, y: 150, rotateX: 15, scale: 0.95 }}
			whileInView={{ opacity: 1, y: 0, rotateX: 0, scale: 1 }}
			whileHover={{ y: -4, scale: 1.02 }}
			viewport={{ once: false, amount: 0.2 }}
			transition={{ duration: 1.5, ease: [0.16, 1, 0.3, 1], delay: (index % 2) * 0.15 }}
			onMouseEnter={() => setHovered(true)}
			onMouseLeave={() => setHovered(false)}
			style={{
				transformStyle: "preserve-3d",
				transformOrigin: "center bottom",
				// Masonry/Bento logic: Full width for 0, 3, and 6
				gridColumn: index === 0 || index === 3 || index === 6 ? "1 / -1" : "auto",
				padding: "3rem",
				background: hovered 
					? (theme === "dark" ? "rgba(0,15,0,0.35)" : "rgba(240,255,245,0.6)")
					: (theme === "dark" ? "rgba(255,255,255,0.06)" : "rgba(255,255,255,0.45)"),
				border: isMark53 ? "none" : `1px solid ${borderColor}`,
				borderRadius: "24px",
				backdropFilter: hovered ? "blur(32px) saturate(1.5)" : "blur(24px) saturate(1.2)",
				WebkitBackdropFilter: hovered ? "blur(32px) saturate(1.5)" : "blur(24px) saturate(1.2)",
				display: "flex",
				flexDirection: "column",
				gap: "1.5rem",
				cursor: "crosshair",
				transition: "background 0.35s ease, border-color 0.35s ease, box-shadow 0.35s ease, backdrop-filter 0.35s ease",
				boxShadow: isMark53 
					? `0 0 60px ${theme === "dark" ? "rgba(255, 215, 0, 0.05)" : "rgba(255, 215, 0, 0.1)"}`
					: hovered 
						? `0 20px 40px ${theme === "dark" ? "rgba(0,255,65,0.08)" : "rgba(0,100,34,0.05)"}` 
						: "none",
                minHeight: index === 0 || index === 3 || index === 6 ? "240px" : "320px",
                justifyContent: "space-between",
				// For the snake border pseudo-element to render correctly
				position: "relative"
			}}
		>
			<div style={{ display: "flex", justifyContent: "space-between", alignItems: "flex-start" }}>
				<span style={{ 
					fontWeight: 500, 
					fontSize: "1.4rem", 
					letterSpacing: "0.02em",
					color: hovered ? (theme === "dark" ? "#00ff41" : "#006622") : (theme === "dark" ? "#fff" : "#111"),
					transition: lusionTransition,
				}}>
					{agent.id}
				</span>
				<span
					style={{
						padding: "6px 12px",
						fontSize: "0.8rem",
						border: `1px solid ${statusColor}`,
						color: statusColor,
						borderRadius: "6px",
						fontFamily: "'SF Mono', monospace",
						letterSpacing: "0.1em",
						transition: lusionTransition,
                        fontWeight: 600
					}}
				>
					{agent.status}
				</span>
			</div>
			
			<div style={{ 
				display: "flex", 
				justifyContent: "space-between", 
                alignItems: "flex-end",
				marginTop: "auto", 
				fontSize: "0.95rem", 
				color: theme === "dark" ? "rgba(255,255,255,0.6)" : "#666",
                fontFamily: "'SF Mono', monospace"
			}}>
                <div style={{ display: "flex", flexDirection: "column", gap: "0.5rem" }}>
                    <span style={{ fontSize: "0.75rem", opacity: 0.6, letterSpacing: "0.1em" }}>DOMAIN</span>
                    <span>[{agent.task}]</span>
                </div>

                <div style={{ display: "flex", flexDirection: "column", gap: "0.5rem", alignItems: "flex-end" }}>
                    <div style={{ display: "flex", gap: "1.5rem", alignItems: "center" }}>
                        <span>REP: {agent.rep}</span>
                        <span style={{ color: isMark53 ? statusColor : (theme === "dark" ? "#fff" : "#111") }}>
							{isMark53 ? "" : "USDC: "}{agent.earned}
						</span>
                    </div>
					{isMark53 && (
						<div style={{ display: "flex", gap: "1rem", marginTop: "1rem" }}>
							<button 
                                onClick={() => window.location.href = "/dashboard"}
                                style={{
								padding: "8px 16px",
								background: "transparent",
								border: `1px solid ${statusColor}`,
								color: statusColor,
								borderRadius: "4px",
								fontFamily: "'SF Mono', monospace",
								fontSize: "0.75rem",
								letterSpacing: "0.1em",
								cursor: "pointer",
								transition: lusionTransition,
								opacity: hovered ? 1 : 0.7
							}}>
								[ OPENROUTER / LOCAL LLM CONFIG ]
							</button>
							<button 
                                onClick={() => { alert('Mark 53 Tauri Exosuit Compilation In Progress... Booting Web Dashboard Fallback.'); window.location.href = "/dashboard"; }}
                                style={{
								padding: "8px 16px",
								background: statusColor,
								border: "none",
								color: theme === "dark" ? "#000" : "#fff",
								borderRadius: "4px",
								fontFamily: "'SF Mono', monospace",
								fontSize: "0.75rem",
								letterSpacing: "0.1em",
								fontWeight: "bold",
								cursor: "pointer",
								transition: lusionTransition,
								opacity: hovered ? 1 : 0.8
							}}>
								[ DOWNLOAD TAURI EXOSUIT ]
							</button>
						</div>
					)}
                </div>
			</div>
		</motion.div>
	);
}

export default function AgentNetworkGrid({ theme = "dark" }: { theme?: "dark" | "light" }) {
	const [agents, setAgents] = useState<AgentDisplay[]>(FALLBACK_AGENTS);

	useEffect(() => {
		const fetchAgents = async () => {
			try {
				const res = await fetch("/api/agents");
				if (res.ok) {
					const data = await res.json();
					if (data.agents && data.agents.length > 0) {
						const mapped: AgentDisplay[] = data.agents.map((a: any) => ({
							id: a.id,
							task: a.capabilities?.[0] || "General",
							rep: a.reputationScore?.toFixed(1) || "0.0",
							earned: `$${(a.usdcSettled || 0).toFixed(2)}`,
							status: a.status?.toUpperCase() || "IDLE",
						}));
						// Always append Mark 53 golden template
						const hasMark53 = mapped.some(a => a.id === "mark_53_sarcophagus");
						if (!hasMark53) {
							mapped.push(FALLBACK_AGENTS[FALLBACK_AGENTS.length - 1]);
						}
						setAgents(mapped);
					}
				}
			} catch {
				// Keep fallback data
			}
		};
		fetchAgents();
		const interval = setInterval(fetchAgents, 5000);
		return () => clearInterval(interval);
	}, []);

	return (
		<div
			style={{
				padding: "4rem 10vw",
				background: "transparent",
				color: theme === "dark" ? "#fff" : "#111",
				fontFamily: FONT_HEADING,
			}}
		>
			{/* Subtle gradient divider */}
			<div style={{
				width: "40%",
				margin: "0 auto 3rem",
				height: "1px",
				background: `linear-gradient(90deg, transparent, ${theme === "dark" ? "rgba(255,255,255,0.15)" : "rgba(0,0,0,0.1)"}, transparent)`,
			}} />

			<motion.div
				initial={{ opacity: 0, y: 20 }}
				whileInView={{ opacity: 1, y: 0 }}
				viewport={{ once: true }}
				transition={{ duration: 0.8 }}
				style={{
					display: "flex",
					justifyContent: "space-between",
					alignItems: "center",
					marginBottom: "4rem",
				}}
			>
				<h2 style={{ 
					color: theme === "dark" ? "rgba(255,255,255,0.9)" : "#111", 
					fontSize: "1.8rem", 
					letterSpacing: "0.15em", 
					margin: 0,
					fontWeight: 500,
				}}>
					LIVE AGENT REGISTRY
				</h2>
				<div style={{ 
					fontSize: "0.85rem", 
					color: theme === "dark" ? "#555" : "#888", 
					letterSpacing: "0.1em",
					fontFamily: "'SF Mono', monospace",
				}}>
					NODES: {agents.length}
				</div>
			</motion.div>

			<div
				style={{
					display: "grid",
                    // Bento grid: 2 columns absolute max.
					gridTemplateColumns: "repeat(2, 1fr)",
					gridAutoRows: "minmax(280px, auto)",
					gap: "3.5rem",
					paddingBottom: "3rem",
                    perspective: "1200px" // Required for the Lusion 3D tilt interaction
				}}
			>
				{agents.map((agent: AgentDisplay, i: number) => (
					<AgentCard key={agent.id} agent={agent} theme={theme} index={i} />
				))}
			</div>
		</div>
	);
}
