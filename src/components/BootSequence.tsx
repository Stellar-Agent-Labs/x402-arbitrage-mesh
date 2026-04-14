"use client";
import React, { useState, useEffect } from "react";
import { motion, AnimatePresence } from "framer-motion";

export default function BootSequence({ onComplete }: { onComplete: () => void }) {
	const [lines, setLines] = useState<string[]>([]);

	useEffect(() => {
		const sequence = [
			"INITIATING BOOT SEQUENCE...",
			"[OK] Kernel Memory Verified",
			"[OK] Extism WASM Sandbox Mounted",
			"[...] Locating P2P Arbitrage Nodes",
			"[OK] Connected to Triarchy Swarm (3 Nodes)",
			"ENGAGING X402 PROTOCOL...",
			"LUSION MATRIX SYNCHRONIZED",
			"ACCESS GRANTED."
		];

		let delay = 0;
		sequence.forEach((line, index) => {
			delay += Math.random() * 300 + 200;
			setTimeout(() => {
				setLines((prev) => [...prev, line]);
				if (index === sequence.length - 1) {
					setTimeout(onComplete, 800);
				}
			}, delay);
		});
	}, [onComplete]);

	return (
		<motion.div
			initial={{ opacity: 1 }}
			exit={{ opacity: 0 }}
			transition={{ duration: 1 }}
			style={{
				position: "fixed",
				top: 0,
				left: 0,
				width: "100vw",
				height: "100vh",
				backgroundColor: "#000",
				color: "#00ff41",
				fontFamily: "monospace",
				padding: "5rem",
				zIndex: 9999,
				overflow: "hidden",
				display: "flex",
				flexDirection: "column",
				justifyContent: "flex-end",
			}}
		>
			{lines.map((line, i) => (
				<motion.div
					key={i}
					initial={{ opacity: 0, x: -20 }}
					animate={{ opacity: 1, x: 0 }}
					style={{ marginBottom: "10px", fontSize: "1.2rem", fontWeight: "bold" }}
				>
					{line}
				</motion.div>
			))}
			<div
				style={{
					height: "20px",
					width: "10px",
					backgroundColor: "#00ff41",
					animation: "blink 1s step-end infinite",
					marginTop: "10px",
				}}
			/>
			<style dangerouslySetInnerHTML={{ __html: "@keyframes blink { 50% { opacity: 0; } }" }} />
		</motion.div>
	);
}
