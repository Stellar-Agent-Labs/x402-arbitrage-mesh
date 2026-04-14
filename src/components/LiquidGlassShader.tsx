"use client";
import { Stars } from "@react-three/drei";
import { Canvas, useFrame } from "@react-three/fiber";
import {
	Bloom,
	ChromaticAberration,
	EffectComposer,
	Noise,
	Vignette,
} from "@react-three/postprocessing";
import { BlendFunction } from "postprocessing";
import { useMemo, useRef, useState } from "react";
import * as THREE from "three";
import RefractiveCore from "./RefractiveCore";

const PARTICLE_COUNT = 3000;

const vertexShader = `
  uniform float uTime;
  attribute vec3 customColor;
  varying vec3 vColor;
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
    
    float phase = uTime * 0.04; 
    float n1 = snoise(pos * 0.35 + phase) * 0.9; 
    float n2 = snoise(pos.yzx * 0.35 + phase + 10.0) * 0.9;
    float n3 = snoise(pos.zxy * 0.35 + phase + 20.0) * 0.9;
    
    vec3 newPos = pos + vec3(n1, n2, n3);
    
    vec4 mvPosition = modelViewMatrix * vec4(newPos, 1.0);
    gl_Position = projectionMatrix * mvPosition;
    gl_PointSize = max(2.5, (55.0 / -mvPosition.z));
  }
`;

function LiquidNebula({ theme }: { theme: "dark" | "light" }) {
	const pointsRef = useRef<THREE.Points>(null);

	const [[positions, colors]] = useState(() => {
		const pos = new Float32Array(PARTICLE_COUNT * 3);
		const col = new Float32Array(PARTICLE_COUNT * 3);
		const baseColor = new THREE.Color("#00ff41");
		const secondaryColor = new THREE.Color("#0fa33a");

		for (let i = 0; i < PARTICLE_COUNT; i++) {
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
			uTheme: { value: theme === "dark" ? 0.0 : 1.0 }, // Pass theme to shader
		}),
		[theme],
	);

    // Update uniform when theme changes
    useFrame(() => {
        if (pointsRef.current) {
            (pointsRef.current.material as THREE.ShaderMaterial).uniforms.uTheme.value = theme === "dark" ? 0.0 : 1.0;
        }
    });

	useFrame((state) => {
		if (!pointsRef.current) return;
		const time = state.clock.elapsedTime;
		(pointsRef.current.material as THREE.ShaderMaterial).uniforms.uTime.value = time;
		pointsRef.current.rotation.y = time * 0.005;
		pointsRef.current.rotation.x = time * 0.002;
	});

    const themedFragmentShader = `
      varying vec3 vColor;
      uniform float uTheme;
      void main() {
        float dist = length(gl_PointCoord - vec2(0.5));
        if (dist > 0.5) discard;
        float alpha = 0.35 * (1.0 - dist * 2.0);
        
        // If theme is light, make particles grey/dark
        vec3 finalColor = mix(vColor, vec3(0.05, 0.05, 0.05), uTheme);
        
        gl_FragColor = vec4(finalColor, alpha);
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
				blending={theme === "dark" ? THREE.AdditiveBlending : THREE.NormalBlending}
			/>
		</points>
	);
}

export default function LiquidGlassShader({ theme = "dark" }: { theme?: "dark" | "light" }) {
	return (
		<div
			style={{
				position: "fixed",
				inset: 0,
				zIndex: -1,
				pointerEvents: "none",
				touchAction: "none",
			}}
		>
			<Canvas dpr={[1, 2]} camera={{ position: [0, 0, 15], fov: 45 }}>
				<color attach="background" args={[theme === "dark" ? "#010201" : "#fafafa"]} />

				{/* Core Lighting for the Refractive Glass */}
				<ambientLight intensity={0.5} color={theme === "dark" ? "#ffffff" : "#cccccc"} />
				
                {theme === "dark" ? (
                    <>
                        <directionalLight position={[10, 10, 10]} intensity={3} color="#00ff41" />
                        <pointLight position={[-10, -10, -10]} intensity={5} color="#0fa33a" />
                    </>
                ) : (
                    <>
                        <directionalLight position={[10, 10, 10]} intensity={1.5} color="#aaaaaa" />
                        <pointLight position={[-10, -10, -10]} intensity={2} color="#cccccc" />
                    </>
                )}

				<Stars
					radius={100}
					depth={50}
					count={6000}
					factor={6}
					saturation={0}
					fade
					speed={3}
				/>
				<LiquidNebula theme={theme} />
				<RefractiveCore />

				{/* Lusion Extreme Post-Processing */}
				<EffectComposer multisampling={4}>
					<Bloom
						luminanceThreshold={theme === "dark" ? 0.2 : 0.8}
						mipmapBlur
						intensity={theme === "dark" ? 1.5 : 0.2}
						blendFunction={theme === "dark" ? BlendFunction.ADD : BlendFunction.MULTIPLY}
					/>
					<ChromaticAberration
						blendFunction={BlendFunction.NORMAL}
						offset={new THREE.Vector2(0.003, 0.003)}
					/>
					<Noise opacity={theme === "dark" ? 0.025 : 0.015} />
					<Vignette eskil={false} offset={0.1} darkness={theme === "dark" ? 1.1 : 0.5} />
				</EffectComposer>
			</Canvas>
		</div>
	);
}
