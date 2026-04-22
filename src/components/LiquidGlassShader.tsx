"use client";
// Stars REMOVED — cannot individually drift, only group rotation
// All particles now unified in LiquidNebula with CPU-side animation
import { Canvas, useFrame, useThree } from "@react-three/fiber";
import {
	Bloom,
	ChromaticAberration,
	EffectComposer,
	Noise,
	SMAA,
} from "@react-three/postprocessing";
import { SMAAPreset } from "postprocessing";
import { BlendFunction } from "postprocessing";
import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { createPortal } from "react-dom";
import * as THREE from "three";
import { useUnifiedPointer } from "../hooks/useUnifiedPointer";
import { useDeviceTier, type DeviceTier } from "../hooks/useDeviceTier";
import RefractiveCore from "./RefractiveCore";
import ScreenPaint from "./ScreenPaint";
import LusionFinalPass from "./LusionFinalPass";
import ScreenPaintDistortion from "./ScreenPaintDistortion";
import BrownianMotionCamera from "./BrownianMotionCamera";
import FsrRcasPass from "./FsrRcasPass";
import LensHaloPass from "./LensHaloPass";

// Lusion-grade adaptive constants per device tier
const TIER_CONFIG = {
	high: { particles: 16384, smaa: SMAAPreset.HIGH, bloomIntensity: 1.5, dpr: [1, 1.5] as [number, number], enableChroma: true },
	mid:  { particles: 8192,  smaa: SMAAPreset.MEDIUM, bloomIntensity: 1.0, dpr: [1, 1.2] as [number, number], enableChroma: true },
	low:  { particles: 4096,  smaa: SMAAPreset.LOW, bloomIntensity: 0.6, dpr: [1, 1] as [number, number], enableChroma: false },
};

// Lusion-grade softness constants (from Blueprint §3, строка 48763)
const U_FOCUS_DIST = 0.32;
const U_P_SOFT_MUL = 0.92;
const U_OPACITY = 0.32;

const vertexShader = `
  uniform float uTime;
  uniform vec2 uResolution;
  attribute vec3 customColor;
  varying vec3 vColor;
  varying float vSoftness;
  varying float vOpacity;

  void main() {
    vColor = customColor;
    vec4 mvPosition = modelViewMatrix * vec4(position, 1.0);
    gl_Position = projectionMatrix * mvPosition;

    // Lusion-exact pSize formula (строка 48602):
    // pSize = (coef * 200.0 * u_pSizeMul) / -mvPosition.z * resolution.y / 1280
    float focusDist = ${U_FOCUS_DIST} * 10.0;
    float coef = abs(-mvPosition.z - focusDist) * 0.3 + pow(max(0.0, -mvPosition.z - focusDist), 2.5) * 0.5;
    float pSize = (coef * 200.0 * 0.4) / max(0.001, -mvPosition.z) * uResolution.y / 1280.0;
    gl_PointSize = max(1.5, pSize);

    // Lusion-exact softness
    vSoftness = coef * ${U_P_SOFT_MUL} * 10.0;

    // Zonal brightness: center=full, edges fade
    float distFromCenter = length(position.xy) / 5.0;
    float isCenter = 1.0 - smoothstep(0.2, 0.6, distFromCenter);
    float edgeFade = 0.35 + 0.65 * (1.0 - smoothstep(0.4, 1.0, distFromCenter));
    vOpacity = ${U_OPACITY} * mix(edgeFade, 1.0, isCenter);
  }
`;

function LiquidNebula({ theme, particleCount }: { theme: "dark" | "light"; particleCount: number }) {
	const pointsRef = useRef<THREE.Points>(null);
	const { size } = useThree();

	// Store base positions + random params ONCE, Lusion-exact spawn bounds
	const [[basePositions, positions, colors, randoms]] = useState(() => {
		const base = new Float32Array(particleCount * 3);
		const pos = new Float32Array(particleCount * 3);
		const col = new Float32Array(particleCount * 3);
		const rnd = new Float32Array(particleCount * 4);
		const baseColor = new THREE.Color("#e8dcc8");
		const secondaryColor = new THREE.Color("#0fa33a");

		// Lusion-exact spawn bounds (строка 48653):
		// bounds: {x:4, y:2.4, z:0.64}, offset: {x:-3, y:-0.5, z:0}
		const BOUNDS = { x: 4, y: 2.4, z: 0.64 };
		const OFFSET = { x: -3, y: -0.5, z: 0 };

		for (let i = 0; i < particleCount; i++) {
			const x = (Math.random() - 0.5) * 2 * BOUNDS.x + OFFSET.x;
			const y = (Math.random() - 0.5) * 2 * BOUNDS.y + OFFSET.y;
			const z = (Math.random() - 0.5) * 2 * BOUNDS.z + OFFSET.z;

			base[i * 3] = x;
			base[i * 3 + 1] = y;
			base[i * 3 + 2] = z;
			pos[i * 3] = x;
			pos[i * 3 + 1] = y;
			pos[i * 3 + 2] = z;

			const c = Math.random() > 0.5 ? baseColor : secondaryColor;
			col[i * 3] = c.r;
			col[i * 3 + 1] = c.g;
			col[i * 3 + 2] = c.b;

			rnd[i * 4] = Math.random() * 6.2832;               // phase (0-2π)
			rnd[i * 4 + 1] = 0.08 + Math.random() * 0.20;      // speed (0.08-0.28) — Lusion curlStr range
			rnd[i * 4 + 2] = 0.3 + Math.random() * 0.8;        // orbit radius (0.3-1.1) — within Lusion bounds
			rnd[i * 4 + 3] = 0.7 + Math.random() * 0.3;        // opacity mul (0.7-1.0)
		}
		return [base, pos, col, rnd];
	});

	const uniforms = useMemo(
		() => ({
			uTime: { value: 0 },
			uTheme: { value: theme === "dark" ? 0.0 : 1.0 },
			uResolution: { value: new THREE.Vector2(size.width, size.height) },
		}),
		[theme, size],
	);

	// CPU-SIDE ANIMATION — guaranteed visible motion
	useFrame((state) => {
		if (!pointsRef.current) return;
		const time = state.clock.elapsedTime;
		(pointsRef.current.material as THREE.ShaderMaterial).uniforms.uTime.value = time;
		(pointsRef.current.material as THREE.ShaderMaterial).uniforms.uTheme.value = theme === "dark" ? 0.0 : 1.0;

		// Animate EVERY particle position each frame
		const posAttr = pointsRef.current.geometry.attributes.position;
		const arr = posAttr.array as Float32Array;

		for (let i = 0; i < particleCount; i++) {
			const phase = randoms[i * 4];
			const speed = randoms[i * 4 + 1];
			const radius = randoms[i * 4 + 2];

			const t = time * speed + phase;
			arr[i * 3]     = basePositions[i * 3]     + Math.sin(t) * radius;
			arr[i * 3 + 1] = basePositions[i * 3 + 1] + Math.cos(t * 1.3 + phase * 2) * radius * 0.7;
			arr[i * 3 + 2] = basePositions[i * 3 + 2] + Math.sin(t * 0.7 + phase * 3) * radius * 0.15;
		}
		posAttr.needsUpdate = true;

		// Slow group rotation for macro drift
		pointsRef.current.rotation.y = time * 0.012;
		pointsRef.current.rotation.x = time * 0.006;
	});

    const themedFragmentShader = `
      // WebGL2: fwidth/dFdx are built-in, no #extension needed
      varying vec3 vColor;
      varying float vSoftness;
      varying float vOpacity;
      uniform float uTheme;

      float linearStep(float edge0, float edge1, float x) {
        return clamp((x - edge0) / (edge1 - edge0), 0.0, 1.0);
      }

      void main() {
        float d = length(gl_PointCoord.xy * 2.0 - 1.0);
        float b = linearStep(0.0, vSoftness + fwidth(d), 1.0 - d);
        vec3 finalColor = mix(vColor, vec3(0.05, 0.05, 0.05), uTheme);
        vec3 color = finalColor * b * vOpacity;
        gl_FragColor = vec4(color, b * vOpacity);
      }
    `;

	return (
		<points ref={pointsRef}>
			<bufferGeometry>
				<bufferAttribute attach="attributes-position" args={[positions, 3]} />
				<bufferAttribute attach="attributes-customColor" args={[colors, 3]} />
			</bufferGeometry>
			<shaderMaterial
				vertexShader={vertexShader}
				fragmentShader={themedFragmentShader}
				uniforms={uniforms}
				transparent
				depthWrite={false}
				depthTest={false}
				blending={theme === "dark" ? THREE.AdditiveBlending : THREE.NormalBlending}
				extensions-derivatives={true}
			/>
		</points>
	);
}

function VoltageLights({ theme }: { theme: "dark" | "light" }) {
	const dirLight = useRef<THREE.DirectionalLight>(null);
	const ptLight = useRef<THREE.PointLight>(null);

	useFrame(() => {
		if (theme === "dark" && dirLight.current && ptLight.current) {
			// Constant stable lighting — no pulsation/flickering
			dirLight.current.intensity = 2.8;
			ptLight.current.intensity = 4.0;
		}
	});

	return (
		<group>
			<ambientLight intensity={0.5} color={theme === "dark" ? "#ffffff" : "#cccccc"} />
			<directionalLight ref={dirLight} position={[10, 10, 10]} intensity={theme === "dark" ? 3 : 1.5} color={theme === "dark" ? "#c8bfae" : "#aaaaaa"} />
			<pointLight ref={ptLight} position={[-10, -10, -10]} intensity={theme === "dark" ? 5 : 2} color={theme === "dark" ? "#0fa33a" : "#cccccc"} />
		</group>
	);
}

/**
 * Adaptive post-processing pipeline — Lusion-grade (Blueprint §FSR + §SMAA)
 * Pipeline order matches Lusion exactly (строка 49553-49555):
 *   Scene → SMAA → FSR RCAS → Bloom → LusionFinalPass(vignette+dither+color) → ScreenPaintDistortion
 *
 * High: Full pipeline (SMAA HIGH + FSR RCAS + Bloom + ChromaticAberration + Noise + LusionFinal + ScreenPaintDistortion)
 * Mid:  Reduced pipeline (SMAA MEDIUM + FSR RCAS + Bloom reduced + LusionFinal + ScreenPaintDistortion)
 * Low:  Minimal pipeline (SMAA LOW + FSR RCAS + LusionFinal only)
 */
function AdaptivePostProcessing({ theme, tier, paintTexture }: { theme: "dark" | "light"; tier: DeviceTier; paintTexture: THREE.Texture | null }) {
	const cfg = TIER_CONFIG[tier];

	if (tier === "low") {
		return (
			<EffectComposer multisampling={0}>
				<SMAA preset={cfg.smaa} />
				<FsrRcasPass sharpness={1.0} />
				<LusionFinalPass theme={theme} tintOpacity={0} vignetteFrom={0.8} vignetteTo={1.8} />
			</EffectComposer>
		);
	}

	if (tier === "mid") {
		return (
			<EffectComposer multisampling={0}>
				<SMAA preset={cfg.smaa} />
				<FsrRcasPass sharpness={1.0} />
				{/* Bloom disabled — causes full overexposure without aggressive vignette */}
				{/* LensHaloPass disabled — creates center overexposure on our scene */}
				<ChromaticAberration
					blendFunction={BlendFunction.NORMAL}
					offset={new THREE.Vector2(0.001, 0.001)}
				/>
				<LusionFinalPass theme={theme} tintOpacity={0} vignetteFrom={0.8} vignetteTo={1.8} />
				{/* ScreenPaintDistortion disabled — too aggressive for our scene */}
			</EffectComposer>
		);
	}

	// tier === "high" — full Lusion pipeline
	return (
		<EffectComposer multisampling={0}>
			<SMAA preset={cfg.smaa} />
			<FsrRcasPass sharpness={1.0} />
			{/* Bloom disabled — causes full overexposure without aggressive vignette */}
			{/* LensHaloPass disabled */}
			<ChromaticAberration
				blendFunction={BlendFunction.NORMAL}
				offset={new THREE.Vector2(0.001, 0.001)}
			/>
			{/* Noise REMOVED — was adding grainy film grain overlay. Lusion uses
		    1-bit dithering in final pass (already in LusionFinalPass), NOT noise overlay */}
			<LusionFinalPass theme={theme} tintOpacity={0} vignetteFrom={0.8} vignetteTo={1.8} />
			{/* ScreenPaintDistortion disabled — too aggressive for our scene */}
		</EffectComposer>
	);
}

export default function LiquidGlassShader({ theme = "dark" }: { theme?: "dark" | "light" }) {
	const pointerRef = useUnifiedPointer();
	const [paintTexture, setPaintTexture] = useState<THREE.Texture | null>(null);
	const tier = useDeviceTier();
	const cfg = TIER_CONFIG[tier];

	const handlePaintTexture = useCallback((tex: THREE.Texture) => {
		setPaintTexture(tex);
	}, []);

	// Portal to body — bypass Lenis CSS transforms that break position:fixed
	const [portalTarget, setPortalTarget] = useState<HTMLElement | null>(null);
	useEffect(() => {
		setPortalTarget(document.body);
	}, []);

	if (!portalTarget) return null;

	return createPortal(
		<div
			style={{
				position: "fixed",
				inset: 0,
				zIndex: -1,
				pointerEvents: "auto",
				touchAction: "none",
			}}
		>
			<Canvas dpr={cfg.dpr} camera={{ position: [0, 0, 15], fov: 45 }}>
				<color attach="background" args={[theme === "dark" ? "#010201" : "#fafafa"]} />
				
				{/* ScreenPaint: Lusion fluid mouse simulation (Blueprint §5) */}
				<ScreenPaint pointerRef={pointerRef} onTextureReady={handlePaintTexture} />

				{/* Core Lighting & Voltage Surges */}
				<VoltageLights theme={theme} />

				{/* Stars REMOVED — drei Stars cannot individually drift */}
				<LiquidNebula theme={theme} particleCount={cfg.particles} />
				
				{/* RefractiveCore: skip on low-tier (saves 5 full scene re-renders) */}
				{tier !== "low" && <RefractiveCore tier={tier} />}

				{/* BrownianMotionCamera permanently disabled:
				    Camera rotation causes parallax between HTML DOM text and 3D objects.
				    No amount of position tracking can fix rotation-induced drift.
				    Particles + stars already provide enough ambient motion. */}

				{/* Adaptive Post-Processing Pipeline — Lusion pipeline order */}
				<AdaptivePostProcessing theme={theme} tier={tier} paintTexture={paintTexture} />
			</Canvas>
		</div>,
		portalTarget,
	);
}
