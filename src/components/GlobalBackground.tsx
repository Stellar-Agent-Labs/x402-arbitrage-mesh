"use client";

import dynamic from "next/dynamic";

// Lazy-load the WebGL particle system to avoid SSR + keep bundle split
const LiquidGlassShader = dynamic(() => import("./LiquidGlassShader"), {
	ssr: false,
});

export default function GlobalBackground() {
	return <LiquidGlassShader theme="dark" />;
}
