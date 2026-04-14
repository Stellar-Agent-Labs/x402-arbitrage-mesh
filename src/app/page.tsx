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

					<AgentNetworkGrid theme={theme} />
				</>
			)}
		</main>
	);
}
