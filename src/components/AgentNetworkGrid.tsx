"use client";
import React, { useState } from "react";
import { motion } from "framer-motion";

const AGENTS = [
	{ id: "x402-AEGIS-NODE", task: "Security Matrix", rep: "99.9", earned: "$1,240.50", status: "ACTIVE" },
	{ id: "agent_alpha_arbitrage", task: "DEX Arbitrage", rep: "95.0", earned: "$420.00", status: "ACTIVE" },
	{ id: "stellar_scrapper_v2", task: "Data Injection", rep: "88.5", earned: "$110.20", status: "IDLE" },
	{ id: "malicious_node_x9", task: "Phishing Attempt", rep: "12.0", earned: "$0.00", status: "QUARANTINED" },
	{ id: "cortex_reviewer", task: "Code Audit", rep: "97.2", earned: "$890.00", status: "ACTIVE" },
	{ id: "liquidity_sniper", task: "Flash Loans", rep: "91.4", earned: "$3,400.10", status: "ACTIVE" },
];

const FONT_HEADING = "'Helvetica Now Display', 'Inter', sans-serif";
const lusionTransition = "all 0.35s cubic-bezier(0.4, 0, 0.2, 1)";

function AgentCard({ agent, theme, index }: { agent: typeof AGENTS[0]; theme: "dark" | "light"; index: number }) {
	const [hovered, setHovered] = useState(false);
	const isQuarantined = agent.status === "QUARANTINED";
	
	const borderColor = isQuarantined 
		? "#ff003c" 
		: hovered 
			? (theme === "dark" ? "rgba(0,255,65,0.5)" : "rgba(0,100,34,0.4)")
			: (theme === "dark" ? "rgba(255,255,255,0.08)" : "rgba(0,0,0,0.1)");
	const bgColor = hovered 
		? (theme === "dark" ? "rgba(0,15,0,0.5)" : "rgba(240,255,245,0.7)")
		: (theme === "dark" ? "rgba(20,20,28,0.4)" : "rgba(255,255,255,0.6)");
	const statusColor = isQuarantined 
		? "#ff003c" 
		: hovered 
			? (theme === "dark" ? "#00ff41" : "#006622")
			: (theme === "dark" ? "rgba(255,255,255,0.6)" : "#333");

	return (
		<motion.div
			initial={{ opacity: 0, y: 50 }}
			whileInView={{ opacity: 1, y: 0 }}
			viewport={{ once: true }}
			transition={{ delay: index * 0.1, duration: 0.5 }}
			onMouseEnter={() => setHovered(true)}
			onMouseLeave={() => setHovered(false)}
			style={{
				padding: "1.5rem",
				background: bgColor,
				border: `1px solid ${borderColor}`,
				borderRadius: "8px",
				backdropFilter: "blur(15px)",
				display: "flex",
				flexDirection: "column",
				gap: "1rem",
				cursor: "crosshair",
				transition: lusionTransition,
				boxShadow: hovered ? `0 0 20px ${theme === "dark" ? "rgba(0,255,65,0.15)" : "rgba(0,100,34,0.1)"}` : "none",
				transform: hovered ? "scale(1.02)" : "scale(1)",
			}}
		>
			<div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
				<span style={{ 
					fontWeight: 500, 
					fontSize: "1.1rem", 
					letterSpacing: "0.03em",
					color: hovered ? (theme === "dark" ? "#00ff41" : "#006622") : (theme === "dark" ? "#fff" : "#111"),
					transition: lusionTransition,
				}}>
					{agent.id}
				</span>
				<span
					style={{
						padding: "2px 8px",
						fontSize: "0.75rem",
						border: `1px solid ${statusColor}`,
						color: statusColor,
						borderRadius: "4px",
						fontFamily: "'SF Mono', monospace",
						letterSpacing: "0.05em",
						transition: lusionTransition,
					}}
				>
					{agent.status}
				</span>
			</div>
			<div style={{ 
				color: theme === "dark" ? "#666" : "#888", 
				fontSize: "0.85rem",
				fontFamily: "'SF Mono', monospace",
			}}>
				[{agent.task}]
			</div>
			
			<div style={{ 
				display: "flex", 
				justifyContent: "space-between", 
				marginTop: "0.5rem", 
				fontSize: "0.85rem", 
				color: theme === "dark" ? "rgba(255,255,255,0.5)" : "#555",
			}}>
				<span>REP: {agent.rep}</span>
				<span>USDC: {agent.earned}</span>
			</div>
		</motion.div>
	);
}

export default function AgentNetworkGrid({ theme = "dark" }: { theme?: "dark" | "light" }) {
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
					marginBottom: "3rem",
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
					NODES: {AGENTS.length}
				</div>
			</motion.div>

			<div
				style={{
					display: "grid",
					gridTemplateColumns: "repeat(auto-fit, minmax(300px, 1fr))",
					gap: "1.5rem",
					paddingBottom: "3rem"
				}}
			>
				{AGENTS.map((agent, i) => (
					<AgentCard key={agent.id} agent={agent} theme={theme} index={i} />
				))}
			</div>
		</div>
	);
}
