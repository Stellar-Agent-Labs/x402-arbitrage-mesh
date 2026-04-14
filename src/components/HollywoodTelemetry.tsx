"use client";
import React, { useEffect, useState } from "react";

export default function HollywoodTelemetry({ theme = "dark" }: { theme?: "dark" | "light" }) {
	const [logs, setLogs] = useState<{ id: string | number; text: string }[]>([]);

	// Simulate system initialization telemetry
	useEffect(() => {
		const bootLogs = [
			"INIT_SEQUENCE_START",
			"MOUNTING_LOCAL_FILESYSTEM...",
			"ESTABLISHING_SECURE_TUNNEL_TO_L402...",
			"WASM_SANDBOX_ALLOCATION: 1024MB",
			"BYPASSING_EXTERNAL_FIREWALLS",
			"AUTH_PROTOCOL: EXTEMPORANEOUS",
			"SYSTEM_READY",
		];
		
		const liveTelemetryNodes = [
			"PING_EXTERNAL_NODE_: ",
			"L402_MEMPOOL_SYNC: ",
			"PACKET_INSPECTED_OK: 0x",
			"QUARANTINE_THREAD_SLEEP: ",
			"WASI_0.2_HEARTBEAT: OK",
			"P2P_NODE_DISCOVERY: SCANNING",
			"SOROBAN_RPC_LATENCY: ",
			"EXTISM_PLUGIN_LIFECYCLE: ",
			"SOVEREIGN_ROUTING_HOP: "
		];

		let i = 0;
		let timeoutId: NodeJS.Timeout;

		const streamLogs = () => {
			if (i < bootLogs.length) {
				setLogs((prev) => [
					...prev.slice(-6),
					{ id: Date.now() + i, text: bootLogs[i] },
				]);
				i++;
				timeoutId = setTimeout(streamLogs, 300);
			} else {
				// Infinite dynamic stream
				const randomLog = liveTelemetryNodes[Math.floor(Math.random() * liveTelemetryNodes.length)];
				const suffix = randomLog.endsWith(": ") ? Math.floor(Math.random() * 9999).toString(16).toUpperCase() + "ms" : "";
				
				setLogs((prev) => {
					const newArray = [...prev, { id: Date.now(), text: randomLog + suffix }];
					return newArray.length > 7 ? newArray.slice(newArray.length - 7) : newArray;
				});
				timeoutId = setTimeout(streamLogs, 800 + Math.random() * 2000);
			}
		};

		timeoutId = setTimeout(streamLogs, 300);
		return () => clearTimeout(timeoutId);
	}, []);

	return (
		<div
			style={{
				position: "absolute", // Removed fixed width and made scale dynamic or smaller
				bottom: 40,
				left: 40,
				width: 450, // Reduced from 600
				height: 250, // Reduced from 400
				background: theme === "dark" ? "rgba(0, 10, 0, 0.85)" : "rgba(255, 255, 255, 0.85)",
				border: `1px solid ${theme === "dark" ? "#00ff41" : "rgba(0,0,0,0.2)"}`,
				borderRadius: 4,
				padding: 15, // Reduced padding
				fontFamily: "monospace",
				color: theme === "dark" ? "#00ff41" : "#111",
				fontSize: 11, // Reduced font size
				overflow: "hidden",
				boxShadow: theme === "dark" ? "0 0 25px rgba(0, 255, 65, 0.3)" : "0 0 25px rgba(0, 0, 0, 0.1)",
				backdropFilter: "blur(10px)",
				zIndex: 20
			}}
		>
			<div
				style={{
					borderBottom: `1px solid ${theme === "dark" ? "#00ff41" : "rgba(0,0,0,0.2)"}`,
					paddingBottom: 5,
					marginBottom: 10,
					fontSize: 12, // Reduced
					fontWeight: "bold",
				}}
			>
				&gt; TRIARCHY_SYS_OP // [root@triarchy-gateway]
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
