"use client";
import React, { useEffect, useState } from "react";

const MESSAGES = [
	"L1_BOOT: Extracting Extism WASM parameters...",
	"[EXTISM] Executing Soroban Plugin <x402_gateway.wasm>",
	"WARNING: Unsigned Memory Access Detected - Blocking.",
	"Soroban L402 Signature Verified.",
	"Agent Authorization Context -> ZEGION [GRANTED]",
	"[SYNAPSE POKE] -> target: DIABLO, cmd: RUN_MERCENARY",
	"Aegis Dome Ingress -> 0 Threats.",
	"D2095 Matrix Synchronization Complete.",
	"Farcaster Frame Access Token Mapped.",
];

const GENERATE_ID = () => Math.random().toString(36).substring(2, 9);

export default function HollywoodTelemetry() {
	const [logs, setLogs] = useState<{ id: string; text: string }[]>([]);

	useEffect(() => {
		const interval = setInterval(() => {
			const msg = MESSAGES[Math.floor(Math.random() * MESSAGES.length)];
			setLogs((prev) => {
				const newLogs = [
					...prev,
					{ id: GENERATE_ID(), text: `[${new Date().toISOString()}] ${msg}` },
				];
				return newLogs.slice(-15);
			});
		}, 850);
		return () => clearInterval(interval);
	}, []);

	return (
		<div
			style={{
				position: "absolute",
				bottom: 40,
				left: 40,
				width: 600,
				height: 400,
				background: "rgba(0, 10, 0, 0.85)",
				border: "1px solid #00ff41",
				borderRadius: 4,
				padding: 20,
				fontFamily: "monospace",
				color: "#00ff41",
				fontSize: 14,
				overflow: "hidden",
				boxShadow: "0 0 25px rgba(0, 255, 65, 0.3)",
				backdropFilter: "blur(10px)",
			}}
		>
			<div
				style={{
					borderBottom: "1px solid #00ff41",
					paddingBottom: 10,
					marginBottom: 15,
					fontWeight: "bold",
				}}
			>
				x402-AEGIS-TELEMETRY // [root@triarchy-gateway]
			</div>
			<div style={{ display: "flex", flexDirection: "column", gap: 8 }}>
				{logs.map((logObj, i) => (
					<div
						key={logObj.id}
						style={{ opacity: i === logs.length - 1 ? 1 : 0.6 }}
					>
						{logObj.text}
					</div>
				))}
			</div>
		</div>
	);
}
