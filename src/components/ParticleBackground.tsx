"use client";

import { useMemo, useRef } from 'react';
import { Canvas, useFrame } from '@react-three/fiber';
import { Line, Html } from '@react-three/drei';
import * as THREE from 'three';

const nodeCenters = [
	new THREE.Vector3(-4.5, 4.5, 0), // HTTP 
	new THREE.Vector3(4.5, 4.5, 0),  // Farcaster 
	new THREE.Vector3(0, 2.25, 0),   // Gateway 
	new THREE.Vector3(0, -0.75, 1),  // WASM Sandbox Z-elevated
	new THREE.Vector3(-4.5, -3.75, 0),// Tier 1 
	new THREE.Vector3(0, -3.75, 0),   // Tier 2 
	new THREE.Vector3(4.5, -3.75, 0), // Tier 3 
];

const labels = [
	"AI Agent (HTTP)", "Farcaster Gate", "Sovereign Gateway", 
	"WASM Sandbox", "Tier 1: Local LLM", "Tier 2: Enterprise", "Tier 3: P2P"
];

const vertexShader = `
    attribute vec3 aTarget;
    uniform float time;
    uniform float uScroll;
    varying vec2 vUv;
    varying float vDistance;

    void main() {
      vUv = uv;
      vec3 animatedTarget = aTarget;
      animatedTarget.y += sin(time + aTarget.x) * 0.2;
      
      float easeScroll = uScroll * uScroll * (3.0 - 2.0 * uScroll);
      vec3 finalPos = mix(position, animatedTarget, easeScroll);
      
      float angle = time * 0.1 * (1.0 - easeScroll);
      mat2 rot = mat2(cos(angle), -sin(angle), sin(angle), cos(angle));
      finalPos.xz = rot * finalPos.xz;

      vec4 mvPosition = modelViewMatrix * vec4(finalPos, 1.0);
      gl_Position = projectionMatrix * mvPosition;
      
      gl_PointSize = (12.0 / -mvPosition.z);
      vDistance = finalPos.z;
    }
`;

const fragmentShader = `
    uniform vec3 uColor;
    varying float vDistance;
    void main() {
      float dist = length(gl_PointCoord - vec2(0.5));
      if (dist > 0.5) discard;
      float alpha = 1.0 - (dist * 2.0);
      gl_FragColor = vec4(uColor, alpha * 0.8);
    }
`;

function GPUPoints({ count = 12000, color = "#00ff41" }) {
	const materialRef = useRef<THREE.ShaderMaterial>(null);

	const geometry = useMemo(() => {
		const pos = new Float32Array(count * 3);
		const tgt = new Float32Array(count * 3);

		for (let i = 0; i < count; i++) {
			const i3 = i * 3;
			
			// Initial chaotic position outside bounds
			pos[i3] = (Math.random() - 0.5) * 40;
			pos[i3 + 1] = (Math.random() - 0.5) * 40;
			pos[i3 + 2] = (Math.random() - 0.5) * 20;

			const nodeIndex = i % 8; // 0-6 are nodes, 7 is scattered noise

			if (nodeIndex < 7) {
				// Form the silhouette! Particles spread around the 3D bounding box of the node square.
				const center = nodeCenters[nodeIndex];
				const spreadX = (Math.random() - 0.5) * 3.5; // width silhouette
				const spreadY = (Math.random() - 0.5) * 1.5; // height silhouette
				const spreadZ = (Math.random() - 0.5) * 0.4;
				
				tgt[i3] = center.x + spreadX;
				tgt[i3 + 1] = center.y + spreadY;
				// Add a subtle curvature (like a CRT screen)
				tgt[i3 + 2] = center.z + Math.abs(spreadX) * 0.2 + spreadZ; 
			} else {
				// General background noise
				tgt[i3] = (Math.random() - 0.5) * 20;
				tgt[i3 + 1] = (Math.random() - 0.5) * 20;
				tgt[i3 + 2] = (Math.random() - 0.5) * 10 - 5;
			}
		}
		
		const geo = new THREE.BufferGeometry();
		geo.setAttribute('position', new THREE.BufferAttribute(pos, 3));
		geo.setAttribute('aTarget', new THREE.BufferAttribute(tgt, 3));
		return geo;
	}, [count]);

	const uniforms = useMemo(() => ({
		time: { value: 0 },
		uScroll: { value: 0 },
		uColor: { value: new THREE.Color(color) }
	}), [color]);

	useFrame((state) => {
		if (materialRef.current) {
			const t = state.clock.elapsedTime;
			materialRef.current.uniforms.time.value = t;
			// Use real DOM canvas bounds to precisely trigger magnetism
			const canvasEl = state.gl.domElement;
			const rect = canvasEl.getBoundingClientRect();
			let progress = 1.0 - (rect.top / window.innerHeight);
			progress = Math.max(0, Math.min(1, progress * 1.5)); 
			
			// Adding microscopic breathing (sine wave flutter) to particles when fully formed
			const pulse = progress === 1 ? Math.sin(t * 2.0) * 0.05 : 0;
			materialRef.current.uniforms.uScroll.value += (progress + pulse - materialRef.current.uniforms.uScroll.value) * 0.05;
		}
	});

	return (
		<points geometry={geometry} frustumCulled={false}>
			<shaderMaterial
				ref={materialRef}
				vertexShader={vertexShader}
				fragmentShader={fragmentShader}
				uniforms={uniforms}
				transparent
				depthWrite={false}
				blending={THREE.AdditiveBlending}
			/>
		</points>
	);
}

function Splines({ color }: { color: string }) {
	const paths = useMemo(() => {
		const curves = [
			[[-4.5, 4.5, 0], [0, 3.375, 1], [0, 2.25, 0]],
			[[4.5, 4.5, 0], [0, 3.375, 1], [0, 2.25, 0]],
			[[0, 2.25, 0], [0, 0.75, 1.5], [0, -0.75, 1]],
			[[0, -0.75, 1], [-2.25, -2.25, 1.5], [-4.5, -3.75, 0]],
			[[0, -0.75, 1], [0, -2.25, 1.5], [0, -3.75, 0]],
			[[0, -0.75, 1], [2.25, -2.25, 1.5], [4.5, -3.75, 0]],
		];
		return curves.map(pts => {
			const curve = new THREE.QuadraticBezierCurve3(
				new THREE.Vector3(...pts[0] as [number,number,number]),
				new THREE.Vector3(...pts[1] as [number,number,number]),
				new THREE.Vector3(...pts[2] as [number,number,number])
			);
			return curve.getPoints(50);
		});
	}, []);

	const impulsesRef = useRef<THREE.Group>(null);
	
	useFrame((state) => {
		const t = state.clock.elapsedTime;
		if (impulsesRef.current) {
			impulsesRef.current.children.forEach((child: any, i: number) => {
				if (child.material && child.material.dashOffset !== undefined) {
					// Physical impulse flowing down the channel
					child.material.dashOffset -= 0.015;
					// Unpredictable disruption (flickering neural logic)
					const disruption = Math.sin(t * 4 + i * 11) * Math.sin(t * 1.5);
					child.material.opacity = disruption > 0 ? 0.9 : 0.05;
				}
			});
		}
	});

	return (
		<group>
			{/* Glass Sheath: Almost invisible, thick outer tube */}
			{paths.map((pts, i) => (
				<Line 
					key={`glass-${i}`} 
					points={pts} 
					color={color} 
					lineWidth={4.5} 
					transparent 
					opacity={0.06} 
				/>
			))}
			
			{/* Core Impulse: Intense dashed light running completely inside the glass tube */}
			<group ref={impulsesRef}>
				{paths.map((pts, i) => (
					<Line 
						key={`impulse-${i}`} 
						points={pts} 
						color={color} 
						lineWidth={1.5} 
						dashed 
						dashScale={8} 
						dashSize={0.6} 
						dashOffset={0} 
						transparent 
						opacity={0.9} 
					/>
				))}
			</group>
		</group>
	);
}

export function ParticleBackground({ theme }: { theme: "light" | "dark" }) {
	const color = theme === "dark" ? "#00ff41" : "#000000";
	const textColor = theme === "dark" ? "#D1FFD7" : "#111"; // Bright neon-mint text for dark mode

	return (
		<div style={{ position: "absolute", top: "10%", left: "50%", transform: "translateX(-50%)", width: "100%", height: "80%", zIndex: 0, pointerEvents: "none", opacity: theme === "dark" ? 0.9 : 0.3 }}>
			<Canvas camera={{ position: [0, 0, 14] }} gl={{ antialias: true, alpha: true }}>
				<ambientLight intensity={0.5} />
				<GPUPoints color={color} count={12000} />
				<Splines color={color} />
				
				{/* WebGL Native DOM Nodes Overlay */}
				{nodeCenters.map((center, i) => (
					<Html key={i} position={center} center zIndexRange={[100, 0]}>
						<div style={{
							color: textColor,
							fontFamily: "monospace",
							fontSize: "13px",
							fontWeight: "bold",
							textTransform: "uppercase",
							letterSpacing: "0.15em",
							whiteSpace: "nowrap",
							textShadow: theme === "dark" ? "0 0 15px rgba(0,255,65,0.8), 0 0 30px rgba(0,255,65,0.4)" : "none",
							pointerEvents: "none",
						}}>
							{labels[i]}
						</div>
					</Html>
				))}
			</Canvas>
		</div>
	);
}
