"use client";
import React, { useState } from "react";
import { AnimatePresence, motion, useScroll, useTransform } from "framer-motion";

function ScrollLogo({ theme }: { theme: "dark" | "light" }) {
	const { scrollY } = useScroll();
	const fontSize = useTransform(scrollY, [0, 200], ["1.5rem", "1.2rem"]);
	const opacityFull = useTransform(scrollY, [0, 100], [1, 0]);
	const opacitySmall = useTransform(scrollY, [100, 200], [0, 1]);

	return (
		<motion.div
			style={{
				position: "fixed",
				top: 50,
				left: 40,
				zIndex: 100,
				display: "flex",
				alignItems: "center",
			}}
		>
			{/* Icon part */}
			<motion.div 
                style={{ 
                    fontWeight: "bold", 
                    fontSize: fontSize, 
                    color: theme === "dark" ? "#00ff41" : "#111", 
                    fontFamily: "monospace",
                    marginRight: 10,
					textShadow: theme === "dark" ? "0 0 10px rgba(0,255,65,0.5)" : "none"
                }}
            >
				⬢
			</motion.div>
			{/* Text part */}
			<motion.div
				style={{
					opacity: opacityFull,
					color: theme === "dark" ? "#fff" : "#333",
					fontFamily: "monospace",
					fontWeight: "bold",
					whiteSpace: "nowrap",
					overflow: "hidden"
				}}
			>
				STELLAR-AGENT-LABS
			</motion.div>
			<motion.div
				style={{
					position: "absolute",
					left: 25,
					opacity: opacitySmall,
					color: theme === "dark" ? "#00ff41" : "#111",
					fontFamily: "monospace",
					fontWeight: "bold",
					whiteSpace: "nowrap",
				}}
			>
				x402
			</motion.div>
		</motion.div>
	);
}

function FloatingConnector({ theme }: { theme: "dark" | "light" }) {
    return (
        <motion.div
            initial={{ y: 100, opacity: 0, x: "-50%" }}
            animate={{ y: 0, opacity: 1, x: "-50%" }}
            transition={{ delay: 2, type: "spring", stiffness: 100 }}
            style={{
                position: "fixed",
                bottom: 40,
                left: "50%",
                zIndex: 100,
                background: theme === "dark" ? "rgba(0,15,0,0.5)" : "rgba(255,255,255,0.6)",
                border: `1px solid ${theme === "dark" ? "#00ff41" : "rgba(0,0,0,0.2)"}`,
                borderRadius: "30px",
                padding: "10px 25px",
                backdropFilter: "blur(15px)",
                display: "flex",
                alignItems: "center",
                gap: "20px",
                boxShadow: theme === "dark" ? "0 0 20px rgba(0,255,65,0.2)" : "0 5px 20px rgba(0,0,0,0.1)"
            }}
        >
            <span style={{ 
                color: theme === "dark" ? "#fff" : "#111", 
                fontFamily: "monospace", 
                fontSize: "0.9rem",
                fontWeight: "bold"
            }}>
                READY TO DEPLOY?
            </span>
            <motion.button
                whileHover={{ scale: 1.05, backgroundColor: theme === "dark" ? "#00ff41" : "#111", color: theme === "dark" ? "#000" : "#fff" }}
                whileTap={{ scale: 0.95 }}
                style={{
                    background: "transparent",
                    color: theme === "dark" ? "#00ff41" : "#111",
                    border: `1px solid ${theme === "dark" ? "#00ff41" : "#111"}`,
                    padding: "8px 16px",
                    borderRadius: "20px",
                    cursor: "pointer",
                    fontFamily: "monospace",
                    fontWeight: "bold",
                    transition: "all 0.2s"
                }}
            >
                CONNECT AGENT
            </motion.button>
        </motion.div>
    );
}

import HollywoodTelemetry from "@/components/HollywoodTelemetry";
import LiquidGlassShader from "@/components/LiquidGlassShader";
import FPSCounter from "@/components/FPSCounter";
import BootSequence from "@/components/BootSequence";
import GlitchText from "@/components/GlitchText";
import AgentNetworkGrid from "@/components/AgentNetworkGrid";
import CustomCursor from "@/components/CustomCursor";

export default function Page() {
	const [booted, setBooted] = useState(false);
	const [theme, setTheme] = useState<"dark" | "light">("dark");

	return (
		<main
			style={{ minHeight: "100vh", position: "relative", overflowX: "hidden", background: "transparent" }}
		>
			<AnimatePresence>
				{!booted && <BootSequence key="boot" onComplete={() => setBooted(true)} />}
			</AnimatePresence>

			{booted && (
				<>
					<CustomCursor theme={theme} />
					<FPSCounter />
					<ScrollLogo theme={theme} />
					<FloatingConnector theme={theme} />

					{/* Theme Switcher Button */}
					<motion.button
						onClick={() => setTheme((t) => (t === "dark" ? "light" : "dark"))}
						initial={{ opacity: 0, y: -20 }}
						animate={{ opacity: 1, y: 0 }}
						transition={{ delay: 1, duration: 1 }}
						whileHover={{ scale: 1.05 }}
						whileTap={{ scale: 0.95 }}
						style={{
							position: "fixed",
							bottom: 40,
							right: 40,
							zIndex: 100,
							padding: "10px 24px",
							background: theme === "dark" ? "rgba(0,15,0,0.4)" : "rgba(255,255,255,0.7)",
							border: `1px solid ${theme === "dark" ? "rgba(0, 255, 65, 0.5)" : "rgba(0, 0, 0, 0.2)"}`,
							color: theme === "dark" ? "#00ff41" : "#111",
							backdropFilter: "blur(10px)",
							cursor: "pointer",
							fontFamily: "monospace",
							borderRadius: "30px",
							fontWeight: "bold",
							letterSpacing: "0.1em",
						}}
					>
						{theme === "dark" ? "◉ DARK" : "◌ LIGHT"}
					</motion.button>

					<div style={{ position: "relative", minHeight: "100vh", width: "100%" }}>
						<LiquidGlassShader theme={theme} />

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
								minHeight: "100vh",
								padding: "2rem",
								paddingBottom: "15rem" // Prevent overlap with absolute telemetry
							}}
						>
							<GlitchText text="SOVEREIGN GATEWAY" theme={theme} />
							
							<motion.p
								initial={{ opacity: 0 }}
								animate={{ opacity: 1 }}
								transition={{ delay: 1.5, duration: 1 }}
								style={{
									color: theme === "dark" ? "#00ff41" : "#333",
									fontFamily: "monospace",
									fontSize: "1.2rem",
									marginTop: "2rem",
									letterSpacing: "0.6em",
									textTransform: "uppercase",
								}}
							>
								x402 protocol / Sovereign Matrix
							</motion.p>

							<motion.div
								whileHover={{ scale: 1.05, boxShadow: theme === "dark" ? "0 0 40px rgba(0,255,65,0.7)" : "0 0 40px rgba(0,0,0,0.15)" }}
								whileTap={{ scale: 0.95 }}
								style={{
									marginTop: "4rem",
									padding: "1.5rem 4rem",
									border: `1px solid ${theme === "dark" ? "rgba(0, 255, 65, 0.5)" : "rgba(0, 0, 0, 0.2)"}`,
									borderRadius: "4px",
									background: theme === "dark" ? "rgba(0,15,0,0.4)" : "rgba(255,255,255,0.6)",
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
											theme === "dark" ? "linear-gradient(90deg, transparent, #00ff41, transparent)" : "linear-gradient(90deg, transparent, rgba(0,0,0,0.3), transparent)",
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
											theme === "dark" ? "linear-gradient(90deg, transparent, #00ff41, transparent)" : "linear-gradient(90deg, transparent, rgba(0,0,0,0.3), transparent)",
									}}
								/>
								<p
									style={{
										color: theme === "dark" ? "#fff" : "#111",
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
								onMouseOver={(e) => (e.currentTarget.style.color = theme === "dark" ? "#00ff41" : "#111")}
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
							<HollywoodTelemetry theme={theme} />
						</motion.div>
					</div>

					{/* SYSTEM ARCHITECTURE SECTION */}
					<section style={{ position: "relative", zIndex: 10, padding: "8rem 2rem", maxWidth: 1200, margin: "0 auto", color: theme === "dark" ? "#fff" : "#111" }}>
						<motion.h2 
							initial={{ opacity: 0, y: 20 }}
							whileInView={{ opacity: 1, y: 0 }}
							viewport={{ once: true }}
							style={{ fontFamily: "monospace", fontSize: "2rem", borderBottom: `1px solid ${theme === "dark" ? "#00ff41" : "#333"}`, paddingBottom: "1rem", marginBottom: "4rem" }}
						>
							[ SYSTEM_ARCHITECTURE ]
						</motion.h2>

						<div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(300px, 1fr))", gap: "2rem" }}>
							{[
								{ title: "L402 Payment Layer", desc: "Autonomous USDC micropayments via Soroban on Stellar Testnet. Zero free-riding for resource-heavy AI inferences.", icon: "💸" },
								{ title: "WASM Quarantine", desc: "Extism WASI 0.2 executing untrusted AI payloads with 0ms cold-start latency. Zero filesystem or network access.", icon: "🛑" },
								{ title: "Sovereign Routing", desc: "3-tier intelligent dispatch (Local LLM → Enterprise Node → P2P Mercenary Delegation) based on bounty value and grid topology.", icon: "🌐" }
							].map((feature, idx) => (
								<motion.div
									key={idx}
									initial={{ opacity: 0, y: 30 }}
									whileInView={{ opacity: 1, y: 0 }}
									viewport={{ once: true }}
									transition={{ delay: idx * 0.2 }}
									whileHover={{ y: -5, boxShadow: theme === "dark" ? "0 10px 40px rgba(0,255,65,0.15)" : "0 10px 40px rgba(0,0,0,0.1)" }}
									style={{
										padding: "2rem",
										border: `1px solid ${theme === "dark" ? "rgba(0, 255, 65, 0.3)" : "rgba(0, 0, 0, 0.15)"}`,
										borderRadius: "8px",
										background: theme === "dark" ? "rgba(0, 15, 0, 0.6)" : "rgba(255, 255, 255, 0.8)",
										backdropFilter: "blur(10px)",
									}}
								>
									<div style={{ fontSize: "2rem", marginBottom: "1rem" }}>{feature.icon}</div>
									<h3 style={{ fontFamily: "monospace", color: theme === "dark" ? "#00ff41" : "#111", fontSize: "1.2rem", marginBottom: "1rem" }}>{feature.title}</h3>
									<p style={{ fontFamily: "sans-serif", lineHeight: 1.6, color: theme === "dark" ? "#aaa" : "#555" }}>{feature.desc}</p>
								</motion.div>
							))}
						</div>
					</section>

					{/* SOVEREIGN MANIFESTO & FOOTER */}
					<footer style={{ position: "relative", zIndex: 10, borderTop: `1px solid ${theme === "dark" ? "rgba(0, 255, 65, 0.3)" : "rgba(0,0,0,0.1)"}`, padding: "6rem 2rem 4rem", marginTop: "4rem", background: theme === "dark" ? "rgba(0,10,0,0.7)" : "rgba(255,255,255,0.7)", backdropFilter: "blur(15px)" }}>
						<div style={{ maxWidth: 1200, margin: "0 auto", textAlign: "center" }}>
							<h2 style={{ fontFamily: "monospace", fontSize: "2.5rem", color: theme === "dark" ? "#00ff41" : "#111", marginBottom: "2rem", letterSpacing: "0.1em" }}>
								THE MESH MANIFESTO
							</h2>
							<p style={{ fontFamily: "sans-serif", fontSize: "1.2rem", lineHeight: 1.8, color: theme === "dark" ? "#ccc" : "#444", maxWidth: 800, margin: "0 auto 4rem" }}>
								Other solutions build single, isolated AI agents holding private keys, blindly trusting inputs, and bleeding liquidity. We built the <strong>Immune System</strong> for the entire AI economy. By becoming a Sovereign Integrator, you align your nodes with zero-trust execution and global liquidity flow. Connect your bots, define your rules, and route your intelligence through the gateway.
							</p>
							
							<motion.a
								whileHover={{ scale: 1.05 }}
								whileTap={{ scale: 0.95 }}
								href="https://github.com/Stellar-Agent-Labs/x402-arbitrage-mesh"
								target="_blank"
								rel="noopener noreferrer"
								style={{
									display: "inline-block",
									padding: "1rem 3rem",
									background: theme === "dark" ? "#00ff41" : "#111",
									color: theme === "dark" ? "#000" : "#fff",
									fontFamily: "monospace",
									fontWeight: "bold",
									textDecoration: "none",
									borderRadius: "4px",
									letterSpacing: "0.1em",
									boxShadow: theme === "dark" ? "0 0 30px rgba(0, 255, 65, 0.4)" : "0 8px 30px rgba(0, 0, 0, 0.3)",
									transition: "opacity 0.2s"
								}}
							>
								[ CONNECT SOVEREIGN NODE ]
							</motion.a>
							
							<div style={{ marginTop: "6rem", fontFamily: "monospace", color: theme === "dark" ? "#555" : "#999", fontSize: "0.9rem" }}>
								© {new Date().getFullYear()} Stellar Agent Labs // x402 Global Routing Grid
							</div>
						</div>
					</footer>

					<AgentNetworkGrid theme={theme} />
				</>
			)}
		</main>
	);
}
