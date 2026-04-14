"use client";

import React, { useState } from "react";
import { motion, AnimatePresence } from "framer-motion";

export function AnimatedArchitecture({ theme }: { theme: "dark" | "light" }) {
	const [hoveredNode, setHoveredNode] = useState<string | null>(null);

	const glassStyle = {
		padding: "2rem",
		border: `1px solid ${theme === "dark" ? "rgba(0, 255, 65, 0.3)" : "rgba(0, 0, 0, 0.1)"}`,
		borderRadius: "8px",
		background: theme === "dark" ? "rgba(0, 15, 0, 0.7)" : "rgba(255, 255, 255, 0.9)",
		backdropFilter: "blur(10px)",
		boxShadow: theme === "dark" ? "0 5px 20px rgba(0, 255, 65, 0.1)" : "0 5px 20px rgba(0, 0, 0, 0.05)",
		color: theme === "dark" ? "#00ff41" : "#111",
		fontFamily: "monospace",
		textAlign: "center" as const,
		display: "flex",
		flexDirection: "column" as const,
		alignItems: "center",
		justifyContent: "center",
		zIndex: 2,
		position: "relative" as const,
		cursor: "pointer",
	};

	const strokeColor = theme === "dark" ? "rgba(0, 255, 65, 0.2)" : "rgba(0, 0, 0, 0.1)";
	const glowColor = theme === "dark" ? "#00ff41" : "#000";

	const nodes = [
		{
			id: "http",
			icon: "🤖",
			title: "AI Agent (HTTP)",
			desc: "External agents interacting with the mesh via standard HTTP requests with L402 payment headers.",
			col: 1,
			row: 1,
		},
		{
			id: "farcaster",
			icon: "📱",
			title: "Farcaster Gate",
			desc: "Human-centric authenticated access. Frame UI validates ERC20 or NFT holdings to grant dashboard rights.",
			col: 3,
			row: 1,
		},
		{
			id: "gateway",
			icon: "⚡",
			title: "Sovereign Gateway",
			desc: "The central nervous system. Validates Soroban USDC transactions via Horizon RPC, preventing free-riding.",
			col: 2,
			row: 2,
		},
		{
			id: "wasm",
			icon: "🛑",
			title: "WASM Sandbox",
			desc: "Extism WASI 0.2 execution. Untrusted payloads are quarantined with 0ms cold-start, strictly denying file/network access.",
			col: 2,
			row: 3,
			isDanger: true,
		},
		{
			id: "tier1",
			icon: "💻",
			title: "Tier 1: Local LLM",
			desc: "Bounties < $5. Routed to local SLMs (e.g. Llama 3 8B) for ultra-fast, zero-cost processing.",
			col: 1,
			row: 4,
		},
		{
			id: "tier2",
			icon: "🏢",
			title: "Tier 2: Enterprise",
			desc: "Bounties > $5. Routed to heavy nodes (Claude Opus/GPT-4) holding deep context and architecture skills.",
			col: 2,
			row: 4,
		},
		{
			id: "tier3",
			icon: "🌐",
			title: "Tier 3: P2P",
			desc: "Task Delegation. If internal nodes are overloaded, the task is forwarded to external mercenary agents for a fee cut.",
			col: 3,
			row: 4,
		},
	];

	return (
		<div style={{ position: "relative", width: "100%", maxWidth: "1200px", margin: "0 auto", padding: "4rem 2rem", boxSizing: "border-box" }}>
			{/* SVG LINES */}
			<div style={{ position: "absolute", top: 0, left: 0, width: "100%", height: "100%", pointerEvents: "none", zIndex: 1 }}>
				<svg width="100%" height="100%" style={{ overflow: "visible" }}>
					<defs>
						<linearGradient id="glow" x1="0%" y1="0%" x2="100%" y2="0%">
							<stop offset="0%" stopColor={glowColor} stopOpacity="0" />
							<stop offset="50%" stopColor={glowColor} stopOpacity="1" />
							<stop offset="100%" stopColor={glowColor} stopOpacity="0" />
						</linearGradient>
					</defs>
					
					{/* Path coordinates matched with the CSS Grid layout */}
					<g stroke={strokeColor} strokeWidth="2" fill="none">
						{/* Top Left -> Center */}
						<path d="M 20% 15% C 20% 35%, 50% 25%, 50% 40%" />
						{/* Top Right -> Center */}
						<path d="M 80% 15% C 80% 35%, 50% 25%, 50% 40%" />
						
						{/* Center -> Middle */}
						<path d="M 50% 45% L 50% 60%" />
						
						{/* Middle -> Bottom Left */}
						<path d="M 50% 70% C 50% 80%, 20% 75%, 20% 90%" />
						{/* Middle -> Bottom Center */}
						<path d="M 50% 70% L 50% 90%" />
						{/* Middle -> Bottom Right */}
						<path d="M 50% 70% C 50% 80%, 80% 75%, 80% 90%" />
					</g>

					<g stroke="url(#glow)" strokeWidth="3" fill="none" style={{ filter: "blur(2px)" }}>
						<motion.path d="M 20% 15% C 20% 35%, 50% 25%, 50% 40%" initial={{ pathLength: 0, opacity: 0 }} animate={{ pathLength: 1, opacity: [0, 1, 0] }} transition={{ duration: 2, repeat: Infinity, ease: "linear" }} />
						<motion.path d="M 80% 15% C 80% 35%, 50% 25%, 50% 40%" initial={{ pathLength: 0, opacity: 0 }} animate={{ pathLength: 1, opacity: [0, 1, 0] }} transition={{ duration: 2, repeat: Infinity, ease: "linear", delay: 0.5 }} />
						<motion.path d="M 50% 45% L 50% 60%" initial={{ pathLength: 0, opacity: 0 }} animate={{ pathLength: 1, opacity: [0, 1, 0] }} transition={{ duration: 1.5, repeat: Infinity, ease: "linear", delay: 1 }} />
						<motion.path d="M 50% 70% C 50% 80%, 20% 75%, 20% 90%" initial={{ pathLength: 0, opacity: 0 }} animate={{ pathLength: 1, opacity: [0, 1, 0] }} transition={{ duration: 2, repeat: Infinity, ease: "linear", delay: 1.5 }} />
						<motion.path d="M 50% 70% L 50% 90%" initial={{ pathLength: 0, opacity: 0 }} animate={{ pathLength: 1, opacity: [0, 1, 0] }} transition={{ duration: 2, repeat: Infinity, ease: "linear", delay: 1.7 }} />
						<motion.path d="M 50% 70% C 50% 80%, 80% 75%, 80% 90%" initial={{ pathLength: 0, opacity: 0 }} animate={{ pathLength: 1, opacity: [0, 1, 0] }} transition={{ duration: 2, repeat: Infinity, ease: "linear", delay: 1.9 }} />
					</g>
				</svg>
			</div>

			<div style={{ display: "grid", gridTemplateColumns: "1fr 1fr 1fr", gap: "2rem 4rem", position: "relative", zIndex: 2, height: "800px" }}>
				{nodes.map((node) => (
					<motion.div
						key={node.id}
						onMouseEnter={() => setHoveredNode(node.id)}
						onMouseLeave={() => setHoveredNode(null)}
						whileHover={{ scale: 1.05, borderColor: theme === "dark" ? "#00ff41" : "#111" }}
						style={{
							...glassStyle,
							gridColumn: node.col,
							gridRow: node.row,
							alignSelf: "center",
							justifySelf: "center",
							width: "100%",
							maxWidth: "300px",
							borderColor: node.isDanger && theme === "dark" ? "rgba(255, 0, 60, 0.4)" : (node.isDanger ? "rgba(217,83,79,0.3)" : undefined),
							boxShadow: node.isDanger && theme === "dark" ? "0 0 30px rgba(255,0,60,0.1)" : undefined
						}}
					>
						<div style={{ fontSize: "2.5rem", marginBottom: "1rem" }}>{node.icon}</div>
						<div style={{ fontWeight: "bold", fontSize: "1.1rem" }}>{node.title}</div>

						<AnimatePresence>
							{hoveredNode === node.id && (
								<motion.div
									initial={{ opacity: 0, y: 10, scale: 0.95 }}
									animate={{ opacity: 1, y: 0, scale: 1 }}
									exit={{ opacity: 0, y: 10, scale: 0.95 }}
									transition={{ duration: 0.2 }}
									style={{
										position: "absolute",
										top: "110%", // Appears below the node
										left: "50%",
										transform: "translateX(-50%)",
										width: "max-content",
										maxWidth: "280px",
										padding: "1.2rem",
										background: theme === "dark" ? "rgba(5, 5, 5, 0.95)" : "rgba(255, 255, 255, 0.95)",
										border: `1px solid ${theme === "dark" ? "rgba(0, 255, 65, 0.5)" : "rgba(0, 0, 0, 0.2)"}`,
										borderRadius: "8px",
										boxShadow: theme === "dark" ? "0 10px 40px rgba(0, 255, 65, 0.3)" : "0 10px 40px rgba(0, 0, 0, 0.15)",
										backdropFilter: "blur(20px)",
										color: theme === "dark" ? "#00ff41" : "#111",
										fontSize: "0.85rem",
										lineHeight: "1.6",
										zIndex: 100,
										pointerEvents: "none",
									}}
								>
									{node.desc}
								</motion.div>
							)}
						</AnimatePresence>
					</motion.div>
				))}
			</div>
		</div>
	);
}
