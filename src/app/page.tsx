"use client";
import React, { useState } from "react";
import { AnimatePresence, motion, useScroll, useTransform } from "framer-motion";

const FONT_HEADING = "'Helvetica Now Display', 'Inter', sans-serif";
const FONT_BODY = "'Inter', 'DM Sans', sans-serif";

// ============================================================
// LUSION HOVER SYSTEM:
// Default = Monochrome premium (white-on-dark)
// Hover  = Original green (#00ff41) bleeds in as "reward"
// ============================================================
const palette = {
	dark: {
		accent: "rgba(255,255,255,0.9)",
		accentHover: "#00ff41",               // 🟢 LUSION: green reveal
		accentMuted: "rgba(255,255,255,0.5)",
		border: "rgba(255,255,255,0.12)",
		borderHover: "rgba(0,255,65,0.4)",    // 🟢
		glassBg: "rgba(20,20,28,0.45)",
		glassBgHover: "rgba(0,15,0,0.5)",     // 🟢
		glow: "rgba(255,255,255,0.06)",
		glowHover: "rgba(0,255,65,0.25)",     // 🟢
		text: "#fff",
		textMuted: "#888",
		btnBg: "#fff",
		btnBgHover: "#00ff41",                // 🟢
		btnText: "#0a0a0a",
	},
	light: {
		accent: "#111",
		accentHover: "#006622",
		accentMuted: "rgba(0,0,0,0.5)",
		border: "rgba(0,0,0,0.12)",
		borderHover: "rgba(0,100,34,0.3)",
		glassBg: "rgba(255,255,255,0.6)",
		glassBgHover: "rgba(240,255,245,0.7)",
		glow: "rgba(0,0,0,0.06)",
		glowHover: "rgba(0,100,34,0.15)",
		text: "#111",
		textMuted: "#666",
		btnBg: "#111",
		btnBgHover: "#006622",
		btnText: "#fff",
	},
};

// CSS transition for all Lusion hover elements
const lusionTransition = "all 0.35s cubic-bezier(0.4, 0, 0.2, 1)";

function ScrollLogo({ theme }: { theme: "dark" | "light" }) {
	const p = palette[theme];
	const { scrollY } = useScroll();
	const fontSize = useTransform(scrollY, [0, 200], ["1.5rem", "1.2rem"]);
	const opacityFull = useTransform(scrollY, [0, 100], [1, 0]);
	const opacitySmall = useTransform(scrollY, [100, 200], [0, 1]);
	const [hovered, setHovered] = useState(false);

	return (
		<motion.div
			style={{
				position: "fixed",
				top: 50,
				left: 40,
				zIndex: 100,
				display: "flex",
				alignItems: "center",
				cursor: "pointer",
			}}
			onMouseEnter={() => setHovered(true)}
			onMouseLeave={() => setHovered(false)}
		>
			<motion.div 
				style={{ 
					fontWeight: 500, 
					fontSize: fontSize, 
					color: hovered ? p.accentHover : p.accent, 
					fontFamily: FONT_HEADING,
					marginRight: 10,
					transition: lusionTransition,
					textShadow: hovered ? `0 0 20px ${p.glowHover}` : "none",
				}}
			>
				⬢
			</motion.div>
			<motion.div
				style={{
					opacity: opacityFull,
					color: hovered ? p.accentHover : p.text,
					fontFamily: FONT_HEADING,
					fontWeight: 500,
					whiteSpace: "nowrap",
					overflow: "hidden",
					letterSpacing: "0.05em",
					transition: lusionTransition,
				}}
			>
				TRIARCHY LABS
			</motion.div>
			<motion.div
				style={{
					position: "absolute",
					left: 25,
					opacity: opacitySmall,
					color: hovered ? p.accentHover : p.accent,
					fontFamily: FONT_HEADING,
					fontWeight: 500,
					whiteSpace: "nowrap",
					transition: lusionTransition,
				}}
			>
				x402
			</motion.div>
		</motion.div>
	);
}

function FloatingConnector({ theme }: { theme: "dark" | "light" }) {
	const p = palette[theme];
	const [hovered, setHovered] = useState(false);
	return (
		<motion.div
			initial={{ y: 50, opacity: 0 }}
			whileInView={{ y: 0, opacity: 1 }}
			viewport={{ once: true }}
			transition={{ delay: 0.2, type: "spring", stiffness: 100 }}
			onMouseEnter={() => setHovered(true)}
			onMouseLeave={() => setHovered(false)}
			style={{
				position: "relative",
				margin: "4rem auto 8rem auto",
				width: "fit-content",
				background: hovered ? p.glassBgHover : p.glassBg,
				border: `1px solid ${hovered ? p.borderHover : p.border}`,
				borderRadius: "30px",
				padding: "10px 25px",
				backdropFilter: "blur(20px)",
				display: "flex",
				alignItems: "center",
				gap: "20px",
				boxShadow: hovered ? `0 0 30px ${p.glowHover}` : `0 8px 32px ${p.glow}`,
				transition: lusionTransition,
			}}
		>
			<span style={{ 
				color: hovered ? p.accentHover : p.text, 
				fontFamily: FONT_HEADING, 
				fontSize: "0.9rem",
				fontWeight: 500,
				letterSpacing: "0.05em",
				transition: lusionTransition,
			}}>
				READY TO DEPLOY?
			</span>
			<motion.button
				whileTap={{ scale: 0.95 }}
				onClick={() => document.getElementById("agent-registry")?.scrollIntoView({ behavior: "smooth" })}
				style={{
					background: hovered ? p.accentHover : "transparent",
					color: hovered ? "#000" : p.accent,
					border: `1px solid ${hovered ? p.accentHover : p.border}`,
					padding: "8px 16px",
					borderRadius: "20px",
					cursor: "pointer",
					fontFamily: FONT_HEADING,
					fontWeight: 500,
					letterSpacing: "0.05em",
					transition: lusionTransition,
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
import { AnimatedArchitecture } from "@/components/AnimatedArchitecture";

export default function Page() {
	const [booted, setBooted] = useState(false);
	const [theme, setTheme] = useState<"dark" | "light">("dark");
	const p = palette[theme];

	// Lusion hover states for inline elements
	const [hoverTheme, setHoverTheme] = useState(false);
	const [hoverGithub, setHoverGithub] = useState(false);
	const [hoverCTA, setHoverCTA] = useState(false);
	const [hoverFooterCTA, setHoverFooterCTA] = useState(false);

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

					{/* Theme Switcher — Lusion hover */}
					<motion.button
						onClick={() => setTheme((t) => (t === "dark" ? "light" : "dark"))}
						initial={{ opacity: 0, y: -20 }}
						animate={{ opacity: 1, y: 0 }}
						transition={{ delay: 1, duration: 1 }}
						onMouseEnter={() => setHoverTheme(true)}
						onMouseLeave={() => setHoverTheme(false)}
						style={{
							position: "fixed",
							bottom: 40,
							right: 40,
							zIndex: 100,
							padding: "10px 24px",
							background: hoverTheme ? p.glassBgHover : p.glassBg,
							border: `1px solid ${hoverTheme ? p.borderHover : p.border}`,
							color: hoverTheme ? p.accentHover : p.accent,
							backdropFilter: "blur(15px)",
							cursor: "pointer",
							fontFamily: FONT_HEADING,
							borderRadius: "30px",
							fontWeight: 500,
							letterSpacing: "0.1em",
							boxShadow: hoverTheme ? `0 0 25px ${p.glowHover}` : "none",
							transition: lusionTransition,
						}}
					>
						{theme === "dark" ? "◉ DARK" : "◌ LIGHT"}
					</motion.button>
					
					{/* GitHub — Lusion hover + blur */}
					<motion.a
						href="https://github.com/Triarchy-Labs"
						target="_blank"
						rel="noopener noreferrer"
						initial={{ opacity: 0, scale: 0.8 }}
						animate={{ opacity: 1, scale: 1 }}
						transition={{ delay: 1.2, duration: 1 }}
						onMouseEnter={() => setHoverGithub(true)}
						onMouseLeave={() => setHoverGithub(false)}
						style={{
							position: "fixed",
							bottom: 40,
							right: 170,
							zIndex: 100,
							width: 44,
							height: 44,
							display: "flex",
							alignItems: "center",
							justifyContent: "center",
							background: hoverGithub ? p.glassBgHover : p.glassBg,
							border: `1px solid ${hoverGithub ? p.borderHover : p.border}`,
							color: hoverGithub ? p.accentHover : p.accent,
							backdropFilter: "blur(15px)",
							cursor: "pointer",
							borderRadius: "50%",
							boxShadow: hoverGithub ? `0 0 25px ${p.glowHover}` : `0 8px 24px ${p.glow}`,
							transition: lusionTransition,
							transform: hoverGithub ? "scale(1.1) rotate(5deg)" : "scale(1)",
						}}
					>
						<svg height="24" width="24" viewBox="0 0 16 16" fill="currentColor">
							<path d="M8 0C3.58 0 0 3.58 0 8c0 3.54 2.29 6.53 5.47 7.59.4.07.55-.17.55-.38 0-.19-.01-.82-.01-1.49-2.01.37-2.53-.49-2.69-.94-.09-.23-.48-.94-.82-1.13-.28-.15-.68-.52-.01-.53.63-.01 1.08.58 1.23.82.72 1.21 1.87.87 2.33.66.07-.52.28-.87.51-1.07-1.78-.2-3.64-.89-3.64-3.95 0-.87.31-1.59.82-2.15-.08-.2-.36-1.02.08-2.12 0 0 .67-.21 2.2.82.64-.18 1.32-.27 2-.27.68 0 1.36.09 2 .27 1.53-1.04 2.2-.82 2.2-.82.44 1.1.16 1.92.08 2.12.51.56.82 1.27.82 2.15 0 3.07-1.87 3.75-3.65 3.95.29.25.54.73.54 1.48 0 1.07-.01 1.93-.01 2.2 0 .21.15.46.55.38A8.013 8.013 0 0016 8c0-4.42-3.58-8-8-8z"></path>
						</svg>
					</motion.a>

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
								paddingBottom: "15rem"
							}}
						>
							<GlitchText text="SOVEREIGN GATEWAY" theme={theme} />
							
							<motion.p
								initial={{ opacity: 0 }}
								animate={{ opacity: 1 }}
								transition={{ delay: 1.5, duration: 1 }}
								style={{
									color: p.accentMuted,
									fontFamily: FONT_HEADING,
									fontSize: "1.2rem",
									marginTop: "2rem",
									letterSpacing: "0.4em",
									textTransform: "uppercase",
									fontWeight: 400,
								}}
							>
								x402 protocol / Sovereign Matrix
							</motion.p>

							{/* MAIN CTA — Lusion hover: white → green bleed */}
							<motion.div
								whileTap={{ scale: 0.95 }}
								onMouseEnter={() => setHoverCTA(true)}
								onMouseLeave={() => setHoverCTA(false)}
								onClick={() => window.open("https://github.com/Triarchy-Labs/x402-arbitrage-mesh", "_blank")}
								style={{
									marginTop: "4rem",
									padding: "1.5rem 4rem",
									border: `1px solid ${hoverCTA ? p.borderHover : p.border}`,
									borderRadius: "4px",
									background: hoverCTA ? p.glassBgHover : p.glassBg,
									backdropFilter: "blur(20px)",
									cursor: "pointer",
									position: "relative",
									overflow: "hidden",
									boxShadow: hoverCTA ? `0 0 40px ${p.glowHover}` : "none",
									transition: lusionTransition,
									transform: hoverCTA ? "scale(1.02)" : "scale(1)",
								}}
							>
								<div
									style={{
										position: "absolute",
										top: 0,
										left: 0,
										right: 0,
										height: "1px",
										background: `linear-gradient(90deg, transparent, ${hoverCTA ? p.accentHover : p.border}, transparent)`,
										transition: lusionTransition,
									}}
								/>
								<div
									style={{
										position: "absolute",
										bottom: 0,
										left: 0,
										right: 0,
										height: "1px",
										background: `linear-gradient(90deg, transparent, ${hoverCTA ? p.accentHover : p.border}, transparent)`,
										transition: lusionTransition,
									}}
								/>
								<p
									style={{
										color: hoverCTA ? p.accentHover : p.text,
										fontFamily: FONT_HEADING,
										textAlign: "center",
										margin: 0,
										fontWeight: 500,
										letterSpacing: "0.15em",
										fontSize: "1.1rem",
										transition: lusionTransition,
										textShadow: hoverCTA ? `0 0 15px ${p.glowHover}` : "none",
									}}
								>
									[ INITIATE L402 HANDSHAKE ]
								</p>
							</motion.div>

							{/* Live stats row — replaces Farcaster link */}
							<motion.div
								initial={{ opacity: 0 }}
								animate={{ opacity: 1 }}
								transition={{ delay: 2, duration: 1 }}
								style={{
									marginTop: "3rem",
									display: "flex",
									gap: "3rem",
									justifyContent: "center",
									fontFamily: "'SF Mono', monospace",
									fontSize: "0.75rem",
									color: p.textMuted,
									letterSpacing: "0.1em",
								}}
							>
								<span>NODES: 6</span>
								<span>LATENCY: 12ms</span>
								<span>UPTIME: 99.97%</span>
							</motion.div>
						</motion.div>

						<motion.div
							initial={{ opacity: 0, x: -50 }}
							animate={{ opacity: 1, x: 0 }}
							transition={{ delay: 2, duration: 1 }}
						>
							<HollywoodTelemetry theme={theme} />
						</motion.div>
					</div>

					{/* SYSTEM ARCHITECTURE */}
					<section style={{ position: "relative", zIndex: 10, width: "100%", padding: "4rem 0" }}>
						<motion.h2 
							initial={{ opacity: 0, y: 20 }}
							whileInView={{ opacity: 1, y: 0 }}
							viewport={{ once: true }}
							style={{ 
								fontFamily: FONT_HEADING, 
								fontSize: "2rem", 
								textAlign: "center", 
								borderBottom: "none", 
								paddingBottom: "1rem", 
								marginBottom: "2rem", 
								color: p.text,
								fontWeight: 500,
								letterSpacing: "0.1em",
							}}
						>
							SYSTEM ARCHITECTURE
						</motion.h2>

						<AnimatedArchitecture theme={theme} />
						<FloatingConnector theme={theme} />
					</section>

					{/* AGENT REGISTRY */}
					<div id="agent-registry">
						<AgentNetworkGrid theme={theme} />
					</div>

					{/* FOOTER — Lusion hover on CTA */}
					<footer style={{ 
						position: "relative", 
						zIndex: 10, 
						padding: "6rem 2rem 4rem", 
						marginTop: "4rem", 
						background: theme === "dark" 
							? "linear-gradient(to bottom, rgba(10,10,15,0.6), rgba(10,10,15,0.95))" 
							: "linear-gradient(to bottom, rgba(250,250,252,0.6), rgba(250,250,252,0.95))",
						backdropFilter: "blur(30px)",
					}}>
						{/* Green glow divider at top */}
						<div style={{
							position: "absolute",
							top: 0,
							left: "20%",
							right: "20%",
							height: "1px",
							background: `linear-gradient(90deg, transparent, ${theme === "dark" ? "rgba(0,255,65,0.3)" : "rgba(0,100,34,0.2)"}, transparent)`,
						}} />
						<div style={{ maxWidth: 1200, margin: "0 auto", textAlign: "center" }}>
							<h2 style={{ 
								fontFamily: FONT_HEADING, 
								fontSize: "2.5rem", 
								color: p.text, 
								marginBottom: "2rem", 
								letterSpacing: "0.08em",
								fontWeight: 500,
							}}>
								THE MESH MANIFESTO
							</h2>
							<p style={{ 
								fontFamily: FONT_BODY, 
								fontSize: "1.2rem", 
								lineHeight: 1.8, 
								color: p.textMuted, 
								maxWidth: 800, 
								margin: "0 auto 4rem" 
							}}>
								Other solutions build single, isolated AI agents holding private keys, blindly trusting inputs, and bleeding liquidity. We built the <strong>Immune System</strong> for the entire AI economy. By becoming a Sovereign Integrator, you align your nodes with zero-trust execution and global liquidity flow.
							</p>
							
							<motion.a
								whileTap={{ scale: 0.95 }}
								href="https://github.com/Triarchy-Labs/x402-arbitrage-mesh"
								target="_blank"
								rel="noopener noreferrer"
								onMouseEnter={() => setHoverFooterCTA(true)}
								onMouseLeave={() => setHoverFooterCTA(false)}
								style={{
									display: "inline-block",
									padding: "1rem 3rem",
									background: hoverFooterCTA ? p.btnBgHover : p.btnBg,
									color: hoverFooterCTA ? "#000" : p.btnText,
									fontFamily: FONT_HEADING,
									fontWeight: 500,
									textDecoration: "none",
									borderRadius: "4px",
									letterSpacing: "0.1em",
									boxShadow: hoverFooterCTA ? `0 0 40px ${p.glowHover}` : `0 8px 30px ${p.glow}`,
									transition: lusionTransition,
									transform: hoverFooterCTA ? "scale(1.05)" : "scale(1)",
								}}
							>
								CONNECT SOVEREIGN NODE
							</motion.a>
							
							<div style={{ 
								marginTop: "6rem", 
								fontFamily: FONT_BODY, 
								color: p.textMuted, 
								fontSize: "0.9rem" 
							}}>
								© {new Date().getFullYear()} Triarchy Labs // x402 Global Routing Grid
							</div>
						</div>
					</footer>
				</>
			)}
		</main>
	);
}
