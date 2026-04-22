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
  attribute vec4 aRandom; // x=phase(0-2π), y=speed(0.08-0.28), z=orbitR(0.3-1.1), w=opacityMul
  varying vec3 vColor;
  varying float vSoftness;
  varying float vOpacity;

  void main() {
    vColor = customColor;

    // GPU-SIDE orbital drift — Lusion curlStr-equivalent
    float phase = aRandom.x;
    float speed = aRandom.y;
    float orbitR = aRandom.z;
    float t = uTime * speed + phase;

    vec3 drift = vec3(
      sin(t) * orbitR,
      cos(t * 1.3 + phase * 2.0) * orbitR * 0.7,
      sin(t * 0.7 + phase * 3.0) * orbitR * 0.15
    );

    vec3 pos = position + drift;
    vec4 mvPosition = modelViewMatrix * vec4(pos, 1.0);
    gl_Position = projectionMatrix * mvPosition;

    // pSize scaled for our camera z=15 (Lusion at z≈5, so /3 compensation)
    float focusDist = ${U_FOCUS_DIST} * 10.0;
    float coef = abs(-mvPosition.z - focusDist) * 0.3 + pow(max(0.0, -mvPosition.z - focusDist), 2.5) * 0.5;
    float pSize = (coef * 200.0 * 0.06) / max(0.001, -mvPosition.z) * uResolution.y / 1280.0;
    gl_PointSize = max(1.5, pSize);

    // Lusion-exact softness
    vSoftness = coef * ${U_P_SOFT_MUL} * 10.0;

    // Zonal brightness: center=full, edges fade
    float distFromCenter = length(pos.xy) / 14.0;
    float isCenter = 1.0 - smoothstep(0.2, 0.6, distFromCenter);
    float edgeFade = 0.35 + 0.65 * (1.0 - smoothstep(0.4, 1.0, distFromCenter));
    vOpacity = ${U_OPACITY} * mix(edgeFade, 1.0, isCenter) * aRandom.w;
  }
`;

function LiquidNebula({ theme, particleCount }: { theme: "dark" | "light"; particleCount: number }) {
	const pointsRef = useRef<THREE.Points>(null);
	const { size } = useThree();

	// Positions + random params, Lusion-exact spawn bounds
	const [[positions, colors, randoms]] = useState(() => {
		const pos = new Float32Array(particleCount * 3);
		const col = new Float32Array(particleCount * 3);
		const rnd = new Float32Array(particleCount * 4);
		const baseColor = new THREE.Color("#e8dcc8");
		const secondaryColor = new THREE.Color("#0fa33a");

		// Lusion bounds ×3 for our camera z=15 (Lusion z≈5)
		const BOUNDS = { x: 12, y: 7.2, z: 1.92 };
		const OFFSET = { x: 0, y: 0, z: 0 }; // centered for our scene

		for (let i = 0; i < particleCount; i++) {
			pos[i * 3]     = (Math.random() - 0.5) * 2 * BOUNDS.x + OFFSET.x;
			pos[i * 3 + 1] = (Math.random() - 0.5) * 2 * BOUNDS.y + OFFSET.y;
			pos[i * 3 + 2] = (Math.random() - 0.5) * 2 * BOUNDS.z + OFFSET.z;

			const c = Math.random() > 0.5 ? baseColor : secondaryColor;
			col[i * 3] = c.r;
			col[i * 3 + 1] = c.g;
			col[i * 3 + 2] = c.b;

			rnd[i * 4]     = Math.random() * 6.2832;          // phase (0-2π)
			rnd[i * 4 + 1] = 0.08 + Math.random() * 0.20;     // speed
			rnd[i * 4 + 2] = 0.8 + Math.random() * 2.5;       // orbit radius (scaled for z=15 bounds)
			rnd[i * 4 + 3] = 0.7 + Math.random() * 0.3;       // opacity mul
		}
		return [pos, col, rnd];
	});

	const uniforms = useMemo(
		() => ({
			uTime: { value: 0 },
			uTheme: { value: theme === "dark" ? 0.0 : 1.0 },
			uResolution: { value: new THREE.Vector2(size.width, size.height) },
		}),
		[theme, size],
	);

	// GPU handles all animation — only update uTime
	useFrame((state) => {
		if (!pointsRef.current) return;
		const mat = pointsRef.current.material as THREE.ShaderMaterial;
		mat.uniforms.uTime.value = state.clock.elapsedTime;
		mat.uniforms.uTheme.value = theme === "dark" ? 0.0 : 1.0;
		pointsRef.current.rotation.y = state.clock.elapsedTime * 0.008;
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
				<bufferAttribute attach="attributes-aRandom" args={[randoms, 4]} />
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
