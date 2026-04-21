"use client";
import { Stars } from "@react-three/drei";
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
import { useCallback, useMemo, useRef, useState } from "react";
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
	high: { particles: 3000, stars: 6000, smaa: SMAAPreset.HIGH, bloomIntensity: 1.5, dpr: [1, 1.5] as [number, number], enableChroma: true },
	mid:  { particles: 1500, stars: 3000, smaa: SMAAPreset.MEDIUM, bloomIntensity: 1.0, dpr: [1, 1.2] as [number, number], enableChroma: true },
	low:  { particles: 800,  stars: 1500, smaa: SMAAPreset.LOW, bloomIntensity: 0.6, dpr: [1, 1] as [number, number], enableChroma: false },
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
  vec4 permute(vec4 x){return mod(((x*34.0)+1.0)*x, 289.0);}
  vec4 taylorInvSqrt(vec4 r){return 1.79284291400159 - 0.85373472095314 * r;}
  
  float snoise(vec3 v){ 
    const vec2  C = vec2(1.0/6.0, 1.0/3.0) ;
    const vec4  D = vec4(0.0, 0.5, 1.0, 2.0);
    vec3 i  = floor(v + dot(v, C.yyy));
    vec3 x0 = v - i + dot(i, C.xxx);
    vec3 g = step(x0.yzx, x0.xyz);
    vec3 l = 1.0 - g;
    vec3 i1 = min( g.xyz, l.zxy );
    vec3 i2 = max( g.xyz, l.zxy );
    vec3 x1 = x0 - i1 +  C.xxx;
    vec3 x2 = x0 - i2 + 2.0 * C.xxx;
    vec3 x3 = x0 - 1.0 + 3.0 * C.xxx;
    i = mod(i, 289.0);
    vec4 p = permute(permute(permute(
              i.z + vec4(0.0, i1.z, i2.z, 1.0))
            + i.y + vec4(0.0, i1.y, i2.y, 1.0))
            + i.x + vec4(0.0, i1.x, i2.x, 1.0));
    float n_ = 1.0/7.0; 
    vec3  ns = n_ * D.wyz - D.xzx;
    vec4 j = p - 49.0 * floor(p * ns.z *ns.z); 
    vec4 x_ = floor(j * ns.z);
    vec4 y_ = floor(j - 7.0 * x_ );  
    vec4 x = x_ *ns.x + ns.yyyy;
    vec4 y = y_ *ns.x + ns.yyyy;
    vec4 h = 1.0 - abs(x) - abs(y);
    vec4 b0 = vec4(x.xy, y.xy);
    vec4 b1 = vec4(x.zw, y.zw);
    vec4 s0 = floor(b0)*2.0 + 1.0;
    vec4 s1 = floor(b1)*2.0 + 1.0;
    vec4 sh = -step(h, vec4(0.0));
    vec4 a0 = b0.xzyw + s0.xzyw*sh.xxyy;
    vec4 a1 = b1.xzyw + s1.xzyw*sh.zzww;
    vec3 p0 = vec3(a0.xy,h.x);
    vec3 p1 = vec3(a0.zw,h.y);
    vec3 p2 = vec3(a1.xy,h.z);
    vec3 p3 = vec3(a1.zw,h.w);
    vec4 norm = taylorInvSqrt(vec4(dot(p0,p0), dot(p1,p1), dot(p2, p2), dot(p3,p3)));
    p0 *= norm.x; p1 *= norm.y; p2 *= norm.z; p3 *= norm.w;
    vec4 m = max(0.6 - vec4(dot(x0,x0), dot(x1,x1), dot(x2,x2), dot(x3,x3)), 0.0);
    m = m * m;
    return 42.0 * dot( m*m, vec4(dot(p0,x0), dot(p1,x1), dot(p2,x2), dot(p3,x3)));
  }

  void main() {
    vColor = customColor;
    vec3 pos = position;
    
    float phase = uTime * 0.10; // visible drift speed
    float n1 = snoise(pos * 0.35 + phase) * 0.9; 
    float n2 = snoise(pos.yzx * 0.35 + phase + 10.0) * 0.9;
    float n3 = snoise(pos.zxy * 0.35 + phase + 20.0) * 0.9;
    
    vec3 newPos = pos + vec3(n1, n2, n3);
    
    vec4 mvPosition = modelViewMatrix * vec4(newPos, 1.0);
    gl_Position = projectionMatrix * mvPosition;
    float baseSize = max(2.5, (55.0 / -mvPosition.z));
    
    // Lusion proportional scaling: particles scale with screen height
    gl_PointSize = baseSize * max(0.5, (uResolution.y / 1280.0));

    // Lusion depth-aware softness (Blueprint §3, строки 48602-48164)
    float focusDist = ${U_FOCUS_DIST} * 10.0;
    float coef = abs(-mvPosition.z - focusDist) * 0.3 + pow(max(0.0, -mvPosition.z - focusDist), 2.5) * 0.5;
    vSoftness = coef * ${U_P_SOFT_MUL} * 10.0;
    vOpacity = ${U_OPACITY};
  }
`;

function LiquidNebula({ theme, particleCount }: { theme: "dark" | "light"; particleCount: number }) {
	const pointsRef = useRef<THREE.Points>(null);
	const { size } = useThree();

	const [[positions, colors]] = useState(() => {
		const pos = new Float32Array(particleCount * 3);
		const col = new Float32Array(particleCount * 3);
		const baseColor = new THREE.Color("#e8dcc8");
		const secondaryColor = new THREE.Color("#0fa33a");

		for (let i = 0; i < particleCount; i++) {
			const theta = Math.random() * 2 * Math.PI;
			const v = Math.random();
			const phi = Math.acos(2 * v - 1);
			const r = 10 * Math.random() ** 0.5;

			pos[i * 3] = r * Math.sin(phi) * Math.cos(theta);
			pos[i * 3 + 1] = r * Math.sin(phi) * Math.sin(theta);
			pos[i * 3 + 2] = r * Math.cos(phi);

			const c = Math.random() > 0.5 ? baseColor : secondaryColor;
			col[i * 3] = c.r;
			col[i * 3 + 1] = c.g;
			col[i * 3 + 2] = c.b;
		}
		return [pos, col];
	});

	const uniforms = useMemo(
		() => ({
			uTime: { value: 0 },
			uTheme: { value: theme === "dark" ? 0.0 : 1.0 },
			uResolution: { value: new THREE.Vector2(size.width, size.height) },
		}),
		[theme, size],
	);

    useFrame(() => {
        if (pointsRef.current) {
            (pointsRef.current.material as THREE.ShaderMaterial).uniforms.uTheme.value = theme === "dark" ? 0.0 : 1.0;
        }
    });

	useFrame((state) => {
		if (!pointsRef.current) return;
		const time = state.clock.elapsedTime;
		(pointsRef.current.material as THREE.ShaderMaterial).uniforms.uTime.value = time;
		pointsRef.current.rotation.y = time * 0.015;
		pointsRef.current.rotation.x = time * 0.008;
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
			// Subtle breathing — no harsh spikes (no stroboscopic pulsation)
			const isSurge = Math.random() > 0.995; // much rarer surges
			const voltage = isSurge ? Math.random() * 3 + 3 : Math.random() * 0.3 + 2.5;
			dirLight.current.intensity = THREE.MathUtils.lerp(dirLight.current.intensity, voltage, 0.08);
			ptLight.current.intensity = THREE.MathUtils.lerp(ptLight.current.intensity, voltage + 1, 0.08);
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
				<LusionFinalPass theme={theme} />
			</EffectComposer>
		);
	}

	if (tier === "mid") {
		return (
			<EffectComposer multisampling={0}>
				<SMAA preset={cfg.smaa} />
				<FsrRcasPass sharpness={1.0} />
				<Bloom
					luminanceThreshold={theme === "dark" ? 0.2 : 0.8}
					mipmapBlur
					intensity={theme === "dark" ? cfg.bloomIntensity : 0.2}
					blendFunction={theme === "dark" ? BlendFunction.ADD : BlendFunction.MULTIPLY}
				/>
				<LensHaloPass haloWidth={0.12} haloRGBShift={0.02} haloStrength={0.08} />
				<ChromaticAberration
					blendFunction={BlendFunction.NORMAL}
					offset={new THREE.Vector2(0.003, 0.003)}
				/>
				<LusionFinalPass theme={theme} />
				<ScreenPaintDistortion paintTexture={paintTexture} amount={4} multiplier={1} colorMultiplier={2} shade={0.3} />
			</EffectComposer>
		);
	}

	// tier === "high" — full Lusion pipeline
	return (
		<EffectComposer multisampling={0}>
			<SMAA preset={cfg.smaa} />
			<FsrRcasPass sharpness={1.0} />
			<Bloom
				luminanceThreshold={theme === "dark" ? 0.2 : 0.8}
				mipmapBlur
				intensity={theme === "dark" ? cfg.bloomIntensity : 0.2}
				blendFunction={theme === "dark" ? BlendFunction.ADD : BlendFunction.MULTIPLY}
			/>
			<LensHaloPass />
			<ChromaticAberration
				blendFunction={BlendFunction.NORMAL}
				offset={new THREE.Vector2(0.003, 0.003)}
			/>
			<Noise opacity={theme === "dark" ? 0.025 : 0.015} />
			<LusionFinalPass theme={theme} />
			<ScreenPaintDistortion paintTexture={paintTexture} amount={4} multiplier={1} colorMultiplier={2} shade={0.3} />
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

	return (
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

				<Stars
					radius={100}
					depth={50}
					count={cfg.stars}
					factor={6}
					saturation={0}
					fade
					speed={3}
				/>
				<LiquidNebula theme={theme} particleCount={cfg.particles} />
				
				{/* RefractiveCore: skip on low-tier (saves 5 full scene re-renders) */}
				{tier !== "low" && <RefractiveCore tier={tier} />}

				{/* Lusion BrownianMotion camera shake (строка 48928-49034) */}
				{/* Camera shake: slowed 20% from Lusion defaults per Creator feedback */}
				<BrownianMotionCamera positionSpeed={0.096} rotationSpeed={0.24} />

				{/* Adaptive Post-Processing Pipeline — Lusion pipeline order */}
				<AdaptivePostProcessing theme={theme} tier={tier} paintTexture={paintTexture} />
			</Canvas>
		</div>
	);
}
