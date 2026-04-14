"use client";

import React, { useState } from "react";
import { motion, AnimatePresence } from "framer-motion";

// Custom Cyberpunk Icons
const IconHTTP = () => (
	<svg viewBox="0 0 24 24" width="32" height="32" stroke="currentColor" strokeWidth="1.5" fill="none" strokeLinecap="square">
		<path d="M4 8h16M4 16h16M8 4v16M16 4v16" strokeOpacity="0.4" />
		<rect x="6" y="6" width="12" height="12" />
		<circle cx="12" cy="12" r="3" fill="currentColor" />
	</svg>
);
const IconFarcaster = () => (
	<svg viewBox="0 0 24 24" width="32" height="32" stroke="currentColor" strokeWidth="1.5" fill="none" strokeLinecap="square">
		<rect x="5" y="4" width="14" height="16" />
		<path d="M9 16h6M12 4v4" />
		<circle cx="12" cy="11" r="2" />
	</svg>
);
const IconGateway = () => (
	<svg viewBox="0 0 24 24" width="40" height="40" stroke="currentColor" strokeWidth="1.5" fill="none" strokeLinecap="square">
		<polygon points="13 2 3 14 12 14 11 22 21 10 12 10 13 2" fill="currentColor" fillOpacity="0.2" />
	</svg>
);
const IconWasm = () => (
	<svg viewBox="0 0 24 24" width="40" height="40" stroke="currentColor" strokeWidth="1.5" fill="none" strokeLinecap="square">
		<rect x="3" y="3" width="18" height="18" />
		<path d="M8 8l4 4-4 4M16 16h-4" />
	</svg>
);
const IconTier1 = () => (
	<svg viewBox="0 0 24 24" width="32" height="32" stroke="currentColor" strokeWidth="1.5" fill="none" strokeLinecap="square">
		<rect x="4" y="4" width="16" height="12" />
		<path d="M8 20h8M12 16v4" />
	</svg>
);
const IconTier2 = () => (
	<svg viewBox="0 0 24 24" width="32" height="32" stroke="currentColor" strokeWidth="1.5" fill="none" strokeLinecap="square">
		<rect x="6" y="2" width="12" height="20" />
		<path d="M10 6h4M10 10h4M10 14h4M10 18h4" />
	</svg>
);
const IconTier3 = () => (
	<svg viewBox="0 0 24 24" width="32" height="32" stroke="currentColor" strokeWidth="1.5" fill="none" strokeLinecap="square">
		<circle cx="12" cy="12" r="10" />
		<path d="M2 12h20M12 2a15.3 15.3 0 014 10 15.3 15.3 0 01-4 10 15.3 15.3 0 01-4-10 15.3 15.3 0 014-10z" />
	</svg>
);

import { ParticleBackground } from "./ParticleBackground";

export function AnimatedArchitecture({ theme }: { theme: "dark" | "light" }) {
	const [hoveredNode, setHoveredNode] = useState<string | null>(null);

	const strokeColor = theme === "dark" ? "rgba(0, 255, 65, 0.4)" : "rgba(0, 0, 0, 0.2)";
	const glowColor = theme === "dark" ? "#00ff41" : "#000";

	const nodes = [
		{
			id: "http",
			icon: <IconHTTP />,
			title: "AI Agent (HTTP)",
			desc: "External agents interacting with the mesh via standard HTTP requests with L402 payment headers.",
			x: 200, y: 100,
		},
		{
			id: "farcaster",
			icon: <IconFarcaster />,
			title: "Farcaster Gate",
			desc: "Human-centric authenticated access. Frame UI validates ERC20 or NFT holdings to grant dashboard rights.",
			x: 800, y: 100,
		},
		{
			id: "gateway",
			icon: <IconGateway />,
			title: "Sovereign Gateway",
			desc: "The central nervous system. Validates Soroban USDC transactions via Horizon RPC, preventing free-riding.",
			x: 500, y: 250,
		},
		{
			id: "wasm",
			icon: <IconWasm />,
			title: "WASM Sandbox",
			desc: "Extism WASI 0.2 execution. Untrusted payloads are quarantined with 0ms cold-start, strictly denying file/network access.",
			x: 500, y: 450,
			isDanger: true,
		},
		{
			id: "tier1",
			icon: <IconTier1 />,
			title: "Tier 1: Local LLM",
			desc: "Bounties < $5. Routed to local SLMs (e.g. Llama 3 8B) for ultra-fast, zero-cost processing.",
			x: 200, y: 650,
		},
		{
			id: "tier2",
			icon: <IconTier2 />,
			title: "Tier 2: Enterprise",
			desc: "Bounties > $5. Routed to heavy nodes (Claude Opus/GPT-4) holding deep context and architecture skills.",
			x: 500, y: 650,
		},
		{
			id: "tier3",
			icon: <IconTier3 />,
			title: "Tier 3: P2P",
			desc: "Task Delegation. If internal nodes are overloaded, the task is forwarded to external mercenary agents for a fee cut.",
			x: 800, y: 650,
		},
	];

	// Extract hovered data
	const activeNode = nodes.find(n => n.id === hoveredNode);

	return (
		<div style={{ position: "relative", width: "100%", maxWidth: "1000px", margin: "0 auto", padding: "2rem 0 6rem" }}>
			
			<ParticleBackground theme={theme} />

			{/* The Info Terminal (Lusion style floating glass pane) */}
			<div style={{ position: "absolute", top: 0, left: "50%", transform: "translateX(-50%)", width: "100%", zIndex: 100, pointerEvents: "none" }}>
				<AnimatePresence>
					{activeNode && (
						<motion.div
							initial={{ opacity: 0, y: -20, filter: "blur(10px)" }}
							animate={{ opacity: 1, y: 0, filter: "blur(0px)" }}
							exit={{ opacity: 0, y: 10, filter: "blur(10px)" }}
							transition={{ type: "spring", stiffness: 300, damping: 25 }}
							style={{
								margin: "0 auto",
								width: "max-content",
								maxWidth: "600px",
								padding: "1.5rem 2.5rem",
								background: theme === "dark" ? "rgba(0, 15, 0, 0.85)" : "rgba(255, 255, 255, 0.9)",
								border: `1px solid ${activeNode.isDanger && theme === "dark" ? "rgba(255,0,60,0.5)" : (theme === "dark" ? "rgba(0, 255, 65, 0.6)" : "rgba(0, 0, 0, 0.2)")}`,
								boxShadow: activeNode.isDanger && theme === "dark" ? "0 20px 60px rgba(255,0,60,0.2)" : (theme === "dark" ? "0 20px 60px rgba(0, 255, 65, 0.2)" : "0 20px 60px rgba(0, 0, 0, 0.1)"),
								backdropFilter: "blur(20px)",
								borderRadius: "12px",
								color: theme === "dark" ? (activeNode.isDanger ? "#ff003c" : "#00ff41") : "#111",
								textAlign: "center",
							}}
						>
							<div style={{ display: "flex", alignItems: "center", justifyContent: "center", gap: "1rem", marginBottom: "0.5rem" }}>
								<div style={{ display: "flex", alignItems: "center", justifyContent: "center", width: "32px", height: "32px", transform: "scale(0.8)" }}>
									{activeNode.icon}
								</div>
								<h3 style={{ fontFamily: "monospace", fontSize: "1.2rem", margin: 0, letterSpacing: "0.1em" }}>{activeNode.title}</h3>
							</div>
							<p style={{ fontFamily: "sans-serif", fontSize: "0.95rem", lineHeight: 1.6, opacity: 0.9, margin: 0 }}>
								{activeNode.desc}
							</p>
						</motion.div>
					)}
				</AnimatePresence>
			</div>

			{/* SVG Coordinate Space */}
			<div style={{ position: "relative", width: "100%", aspectRatio: "1000 / 800", overflow: "visible" }}>
				{/* The Lines */}
				<svg viewBox="0 0 1000 800" style={{ position: "absolute", top: 0, left: 0, width: "100%", height: "100%", pointerEvents: "none", zIndex: 1 }}>
					<defs>
						<linearGradient id="glowLine" x1="0%" y1="0%" x2="100%" y2="0%">
							<stop offset="0%" stopColor={glowColor} stopOpacity="0" />
							<stop offset="50%" stopColor={glowColor} stopOpacity="1" />
							<stop offset="100%" stopColor={glowColor} stopOpacity="0" />
						</linearGradient>
						<filter id="glowBlur" x="-50%" y="-50%" width="200%" height="200%">
							<feGaussianBlur stdDeviation="4" result="blur" />
							<feComposite in="SourceGraphic" in2="blur" operator="over" />
						</filter>
					</defs>
					
					<g stroke={strokeColor} strokeWidth="1" fill="none">
						{/* Paths mapped to exact node coords */}
						<path d="M 200 130 C 200 190, 500 190, 500 220" />
						<path d="M 800 130 C 800 190, 500 190, 500 220" />
						<path d="M 500 280 L 500 420" />
						<path d="M 500 480 C 500 550, 200 550, 200 620" />
						<path d="M 500 480 L 500 620" />
						<path d="M 500 480 C 500 550, 800 550, 800 620" />
					</g>

					<g stroke="url(#glowLine)" strokeWidth="2" fill="none" filter="url(#glowBlur)">
						<motion.path d="M 200 130 C 200 190, 500 190, 500 220" initial={{ pathLength: 0, opacity: 0 }} animate={{ pathLength: 1, opacity: [0, 1, 0] }} transition={{ duration: 2, repeat: Infinity, ease: "linear" }} />
						<motion.path d="M 800 130 C 800 190, 500 190, 500 220" initial={{ pathLength: 0, opacity: 0 }} animate={{ pathLength: 1, opacity: [0, 1, 0] }} transition={{ duration: 2, repeat: Infinity, ease: "linear", delay: 0.5 }} />
						<motion.path d="M 500 280 L 500 420" initial={{ pathLength: 0, opacity: 0 }} animate={{ pathLength: 1, opacity: [0, 1, 0] }} transition={{ duration: 1.5, repeat: Infinity, ease: "linear", delay: 1 }} />
						<motion.path d="M 500 480 C 500 550, 200 550, 200 620" initial={{ pathLength: 0, opacity: 0 }} animate={{ pathLength: 1, opacity: [0, 1, 0] }} transition={{ duration: 2, repeat: Infinity, ease: "linear", delay: 1.5 }} />
						<motion.path d="M 500 480 L 500 620" initial={{ pathLength: 0, opacity: 0 }} animate={{ pathLength: 1, opacity: [0, 1, 0] }} transition={{ duration: 2, repeat: Infinity, ease: "linear", delay: 1.7 }} />
						<motion.path d="M 500 480 C 500 550, 800 550, 800 620" initial={{ pathLength: 0, opacity: 0 }} animate={{ pathLength: 1, opacity: [0, 1, 0] }} transition={{ duration: 2, repeat: Infinity, ease: "linear", delay: 1.9 }} />
					</g>
				</svg>

				{/* The Nodes overlay */}
				{nodes.map(node => (
					<motion.div
						key={node.id}
						onMouseEnter={() => setHoveredNode(node.id)}
						onMouseLeave={() => setHoveredNode(null)}
						whileHover={{ scale: 1.1 }}
						style={{
							position: "absolute",
							left: `${(node.x / 1000) * 100}%`,
							top: `${(node.y / 800) * 100}%`,
							transform: "translate(-50%, -50%)", // perfectly center over coordinate
							zIndex: hoveredNode === node.id ? 20 : 10,
							display: "flex",
							flexDirection: "column",
							alignItems: "center",
							justifyContent: "center",
							cursor: "pointer",
						}}
					>
						{/* Node Box */}
						<div
							style={{
								width: "200px",
								padding: "1.5rem",
								background: theme === "dark" ? "rgba(0, 15, 0, 0.4)" : "rgba(255, 255, 255, 0.6)",
								border: `1px solid ${hoveredNode === node.id ? (node.isDanger && theme === "dark" ? "#ff003c" : strokeColor) : strokeColor}`,
								boxShadow: hoveredNode === node.id ? (node.isDanger && theme === "dark" ? "0 0 30px rgba(255,0,60,0.3)" : (theme === "dark" ? "0 0 30px rgba(0,255,65,0.2)" : "0 5px 20px rgba(0,0,0,0.1)")) : "none",
								backdropFilter: "blur(12px)",
								borderRadius: "4px",
								textAlign: "center",
								color: theme === "dark" ? (node.isDanger ? "#ff003c" : "#00ff41") : "#111",
								transition: "all 0.3s ease",
							}}
						>
							<div style={{ display: "flex", justifyContent: "center", marginBottom: "0.5rem" }}>
								{node.icon}
							</div>
							<div style={{ fontFamily: "monospace", fontSize: "0.9rem", letterSpacing: "0.05em", fontWeight: "bold" }}>
								{node.title}
							</div>
						</div>
					</motion.div>
				))}
			</div>
		</div>
	);
}
