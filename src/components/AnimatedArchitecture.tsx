"use client";

import React from "react";
import { ParticleBackground } from "./ParticleBackground";

export function AnimatedArchitecture({ theme }: { theme: "dark" | "light" }) {
	return (
		<section style={{ padding: "8rem 2rem", position: "relative", zIndex: 10 }}>
			{/* HUD / Header */}
			<div style={{ position: "relative", width: "100%", maxWidth: "1000px", margin: "0 auto", paddingBottom: "2rem", zIndex: 20 }}>
				<div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", borderBottom: `1px solid ${theme === "dark" ? "rgba(0, 255, 65, 0.3)" : "rgba(0,0,0,0.1)"}`, paddingBottom: "1rem" }}>
					<span style={{ fontFamily: "monospace", color: theme === "dark" ? "#00ff41" : "#111", fontSize: "12px", letterSpacing: "0.2em" }}>/// ARCHITECTURE TOPOLOGY</span>
					<div style={{ display: "flex", gap: "10px" }}>
						<div style={{ width: "8px", height: "8px", background: theme === "dark" ? "#00ff41" : "#111", borderRadius: "50%", boxShadow: theme === "dark" ? "0 0 10px #00ff41" : "none" }}></div>
						<div style={{ width: "8px", height: "8px", background: "rgba(255, 0, 60, 0.5)", borderRadius: "50%" }}></div>
						<div style={{ width: "8px", height: "8px", border: `1px solid ${theme === "dark" ? "#00ff41" : "#111"}`, borderRadius: "50%" }}></div>
					</div>
				</div>
			</div>

			{/* NATIVE WEBGL CANVAS CONTAINER */}
			<div style={{ position: "relative", width: "100%", maxWidth: "1200px", height: "800px", margin: "0 auto", overflow: "visible" }}>
				<ParticleBackground theme={theme} />
			</div>
		</section>
	);
}
