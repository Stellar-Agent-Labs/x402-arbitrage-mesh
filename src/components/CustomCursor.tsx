"use client";
import React, { useEffect, useState } from "react";
import { motion } from "framer-motion";

export default function CustomCursor({ theme = "dark" }: { theme?: "dark" | "light" }) {
	const [mousePosition, setMousePosition] = useState({ x: -100, y: -100 });
	const [isHovering, setIsHovering] = useState(false);
	const [isVisible, setIsVisible] = useState(false);

	useEffect(() => {
		const updateMousePosition = (e: MouseEvent) => {
			if (!isVisible) setIsVisible(true);
			setMousePosition({ x: e.clientX, y: e.clientY });
		};

		const handleMouseOver = (e: MouseEvent) => {
			const target = e.target as HTMLElement;
			if (!target) return;
			// Detect interactivity
			const computedStyle = window.getComputedStyle(target);
			if (
				computedStyle.cursor === "pointer" ||
				computedStyle.cursor === "crosshair" ||
				target.closest("button") ||
				target.closest("a")
			) {
				setIsHovering(true);
			} else {
				setIsHovering(false);
			}
		};

		window.addEventListener("mousemove", updateMousePosition);
		window.addEventListener("mouseover", handleMouseOver);
		return () => {
			window.removeEventListener("mousemove", updateMousePosition);
			window.removeEventListener("mouseover", handleMouseOver);
		};
	}, [isVisible]);

	if (!isVisible) return null; // Don't render until mouse moves

	return (
		<motion.div
			style={{
				position: "fixed",
				top: 0,
				left: 0,
				width: 30,
				height: 30,
				borderRadius: "50%",
				backgroundColor: theme === "dark"
					? (isHovering ? "rgba(0, 255, 65, 0.4)" : "rgba(255, 255, 255, 0.2)")
					: (isHovering ? "rgba(0, 0, 0, 0.3)" : "rgba(0, 0, 0, 0.1)"),
				border: isHovering
					? (theme === "dark" ? "1px solid #00ff41" : "1px solid #000")
					: "none",
				pointerEvents: "none",
				zIndex: 99999,
				mixBlendMode: theme === "dark" ? "screen" : "multiply",
			}}
			animate={{
				x: mousePosition.x - 15,
				y: mousePosition.y - 15,
				scale: isHovering ? 2.5 : 1,
			}}
			transition={{
				type: "spring",
				damping: 20, // Low mass, high damping = heavy inertia
				stiffness: 150,
				mass: 0.8,
			}}
		/>
	);
}
