"use client";
import React from "react";
import { motion } from "framer-motion";

const AGENTS = [
	{ id: "x402-AEGIS-NODE", task: "Security Matrix", rep: "99.9", earned: "$1,240.50", status: "ACTIVE" },
	{ id: "agent_alpha_arbitrage", task: "DEX Arbitrage", rep: "95.0", earned: "$420.00", status: "ACTIVE" },
	{ id: "stellar_scrapper_v2", task: "Data Injection", rep: "88.5", earned: "$110.20", status: "IDLE" },
	{ id: "malicious_node_x9", task: "Phishing Attempt", rep: "12.0", earned: "$0.00", status: "QUARANTINED" },
	{ id: "cortex_reviewer", task: "Code Audit", rep: "97.2", earned: "$890.00", status: "ACTIVE" },
	{ id: "liquidity_sniper", task: "Flash Loans", rep: "91.4", earned: "$3,400.10", status: "ACTIVE" },
];

export default function AgentNetworkGrid({ theme = "dark" }: { theme?: "dark" | "light" }) {
	return (
		<div
			style={{
				padding: "4rem 10vw",
				background: theme === "dark" 
					? "linear-gradient(to bottom, transparent, #010a02)" 
					: "linear-gradient(to bottom, transparent, #f0f0f0)",
				color: theme === "dark" ? "#fff" : "#111",
				fontFamily: "monospace",
				borderTop: `1px dashed ${theme === "dark" ? "rgba(0, 255, 65, 0.3)" : "rgba(0,0,0,0.1)"}`,
			}}
		>
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
				<h2 style={{ color: theme === "dark" ? "#00ff41" : "#111", fontSize: "2rem", letterSpacing: "0.2em", margin: 0 }}>
					:: LIVE AGENT REGISTRY
				</h2>
				<div style={{ fontSize: "1rem", color: theme === "dark" ? "#888" : "#555", letterSpacing: "0.1em" }}>
					NETWORK_NODES: {AGENTS.length}
				</div>
			</motion.div>

			<div
				style={{
					display: "grid",
					gridTemplateColumns: "repeat(auto-fit, minmax(300px, 1fr))",
					gap: "2rem",
					paddingBottom: "5rem" // Space for bottom connector
				}}
			>
				{AGENTS.map((agent, i) => (
					<motion.div
						key={agent.id}
						initial={{ opacity: 0, y: 50 }}
						whileInView={{ opacity: 1, y: 0 }}
						viewport={{ once: true }}
						transition={{ delay: i * 0.1, duration: 0.5 }}
						whileHover={{
							scale: 1.02,
							boxShadow: theme === "dark" ? "0 0 20px rgba(0, 255, 65, 0.2)" : "0 0 20px rgba(0,0,0,0.1)",
							borderColor: theme === "dark" ? "#00ff41" : "#000",
						}}
						style={{
							padding: "1.5rem",
							background: theme === "dark" ? "rgba(0, 20, 0, 0.4)" : "rgba(255, 255, 255, 0.6)",
							border: `1px solid ${theme === "dark" ? "rgba(0, 255, 65, 0.1)" : "rgba(0, 0, 0, 0.1)"}`,
							borderRadius: "4px",
							backdropFilter: "blur(10px)",
							display: "flex",
							flexDirection: "column",
							gap: "1rem",
							cursor: "crosshair",
							transition: "border-color 0.3s",
						}}
					>
						<div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
							<span style={{ fontWeight: "bold", fontSize: "1.2rem", letterSpacing: "0.05em" }}>{agent.id}</span>
							<span
								style={{
									padding: "2px 8px",
									fontSize: "0.8rem",
									border: `1px solid ${agent.status === "QUARANTINED" ? "#ff003c" : (theme === "dark" ? "#00ff41" : "#111")}`,
									color: agent.status === "QUARANTINED" ? "#ff003c" : (theme === "dark" ? "#00ff41" : "#111"),
									borderRadius: "2px",
								}}
							>
								{agent.status}
							</span>
						</div>
						<div style={{ color: theme === "dark" ? "#888" : "#555", fontSize: "0.9rem" }}>[{agent.task}]</div>
						
						<div style={{ display: "flex", justifyContent: "space-between", marginTop: "1rem", fontSize: "0.9rem", color: theme === "dark" ? "#00ff41" : "#111" }}>
							<span>REP: {agent.rep}</span>
							<span>USDC: {agent.earned}</span>
						</div>
					</motion.div>
				))}
			</div>
		</div>
	);
}
