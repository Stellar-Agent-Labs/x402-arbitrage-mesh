"use client";
import { useState } from "react";
import { motion, AnimatePresence } from "framer-motion";
import AgentDashboard from "@/components/AgentDashboard";
import HollywoodTelemetry from "@/components/HollywoodTelemetry";
import LiquidGlassShader from "@/components/LiquidGlassShader";
import FPSCounter from "@/components/FPSCounter";
import BootSequence from "@/components/BootSequence";
import GlitchText from "@/components/GlitchText";
import AgentNetworkGrid from "@/components/AgentNetworkGrid";

export default function Page() {
	const [booted, setBooted] = useState(false);

	return (
		<main
			style={{ minHeight: "100vh", position: "relative", overflowX: "hidden", background: "#050505" }}
		>
			<AnimatePresence>
				{!booted && <BootSequence key="boot" onComplete={() => setBooted(true)} />}
			</AnimatePresence>

			{booted && (
				<>
					<FPSCounter />
					<div style={{ position: "relative", height: "100vh", width: "100%" }}>
						<LiquidGlassShader />

						<motion.div
							initial={{ opacity: 0, y: 50 }}
							animate={{ opacity: 1, y: 0 }}
							transition={{ duration: 1.5, ease: "easeOut" }}
							style={{
								position: "relative",
								zIndex: 10,
								display: "flex",
								flexDirection: "column",
								alignItems: "center",
								justifyContent: "center",
								paddingTop: "15vh",
								height: "100%",
								padding: "2rem",
							}}
						>
							<GlitchText text="SOVEREIGN GATEWAY" />
							
							<motion.p
								initial={{ opacity: 0 }}
								animate={{ opacity: 1 }}
								transition={{ delay: 1.5, duration: 1 }}
								style={{
									color: "#00ff41",
									fontFamily: "monospace",
									fontSize: "1.2rem",
									marginTop: "2rem",
									letterSpacing: "0.6em",
									textTransform: "uppercase",
								}}
							>
								x402 protocol / Lusion Matrix
							</motion.p>

							<motion.div
								whileHover={{ scale: 1.05, boxShadow: "0 0 40px rgba(0,255,65,0.7)" }}
								whileTap={{ scale: 0.95 }}
								style={{
									marginTop: "4rem",
									padding: "1.5rem 4rem",
									border: "1px solid rgba(0, 255, 65, 0.5)",
									borderRadius: "4px",
									background: "rgba(0,15,0,0.4)",
									backdropFilter: "blur(15px)",
									cursor: "pointer",
									position: "relative",
									overflow: "hidden",
								}}
							>
								<div
									style={{
										position: "absolute",
										top: 0,
										left: 0,
										right: 0,
										height: "1px",
										background:
											"linear-gradient(90deg, transparent, #00ff41, transparent)",
									}}
								/>
								<div
									style={{
										position: "absolute",
										bottom: 0,
										left: 0,
										right: 0,
										height: "1px",
										background:
											"linear-gradient(90deg, transparent, #00ff41, transparent)",
									}}
								/>
								<p
									style={{
										color: "#fff",
										fontFamily: "monospace",
										textAlign: "center",
										margin: 0,
										fontWeight: "bold",
										letterSpacing: "0.2em",
										fontSize: "1.1rem",
									}}
								>
									[ INITIATE L402 HANDSHAKE ]
								</p>
							</motion.div>

							<a
								href="https://github.com/Stellar-Agent-Labs/farcaster-token-gate"
								target="_blank"
								rel="noopener noreferrer"
								style={{
									marginTop: "2rem",
									color: "#888",
									fontFamily: "monospace",
									textDecoration: "none",
									borderBottom: "1px dashed #888",
									paddingBottom: "2px",
									transition: "color 0.2s",
								}}
								onMouseOver={(e) => (e.currentTarget.style.color = "#00ff41")}
								onMouseOut={(e) => (e.currentTarget.style.color = "#888")}
							>
								Access via Farcaster Frame →
							</a>
						</motion.div>

						<motion.div
							initial={{ opacity: 0, x: -50 }}
							animate={{ opacity: 1, x: 0 }}
							transition={{ delay: 2, duration: 1 }}
						>
							<HollywoodTelemetry />
						</motion.div>

						<AgentDashboard />
					</div>

					<AgentNetworkGrid />
				</>
			)}
		</main>
	);
}
