"use client";
import React, { useState } from "react";
import { motion, useAnimationControls } from "framer-motion";

export default function GlitchText({ text }: { text: string }) {
	const controls = useAnimationControls();
	const [isHovered, setIsHovered] = useState(false);

	const triggerGlitch = async () => {
		if (isHovered) return;
		setIsHovered(true);
		await controls.start({
			x: [0, -5, 5, -2, 2, 0],
			y: [0, 2, -2, 1, -1, 0],
			opacity: [1, 0.5, 1, 0.8, 1],
			filter: [
				"drop-shadow(0 0 40px rgba(0, 255, 65, 0.9))",
				"drop-shadow(-5px 0 0 red) drop-shadow(5px 0 0 blue)",
				"drop-shadow(0 0 40px rgba(0, 255, 65, 0.9))",
			],
			transition: { duration: 0.3, ease: "easeInOut" },
		});
		setIsHovered(false);
	};

	return (
		<motion.div
			style={{ display: "flex", cursor: "crosshair" }}
			onHoverStart={triggerGlitch}
		>
			{text.split("").map((char, index) => (
				<motion.span
					key={index}
					custom={index}
					animate={controls}
					initial={{ opacity: 0, y: 50, rotateX: -90 }}
					transition={{
						opacity: { duration: 0.1, delay: index * 0.05 },
						y: { type: "spring", stiffness: 100, delay: index * 0.05 },
						rotateX: { duration: 0.5, delay: index * 0.05 },
					}}
					style={{
						color: "#fff",
						fontSize: "5rem",
						letterSpacing: char === " " ? "1rem" : "0.15em",
						fontFamily: "monospace",
						textShadow: "0 0 40px rgba(0, 255, 65, 0.9)",
						display: "inline-block",
					}}
				>
					{char}
				</motion.span>
			))}
		</motion.div>
	);
}
