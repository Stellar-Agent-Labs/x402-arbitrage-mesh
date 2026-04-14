"use client";

import { useMemo, useRef } from 'react';
import { Canvas, useFrame } from '@react-three/fiber';
import { Line, Html } from '@react-three/drei';
import * as THREE from 'three';

const nodeCenters = [
	new THREE.Vector3(-4.5, 4.5, 0), // HTTP 
	new THREE.Vector3(4.5, 4.5, 0),  // Farcaster 
	new THREE.Vector3(0, 2.25, 0),   // Gateway 
	new THREE.Vector3(0, -0.75, 1),  // WASM Sandbox Z-elevated (Main Defense Node)
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
    uniform vec3 uNodes[7];
    varying vec2 vUv;
    varying float vDistance;

    // Organic curl noise approximation
    vec3 curlNoise(vec3 p) {
        float x = sin(p.y * 1.5) + cos(p.z * 1.5);
        float y = sin(p.z * 1.5) + cos(p.x * 1.5);
        float z = sin(p.x * 1.5) + cos(p.y * 1.5);
        return vec3(x, y, z) * 0.5;
    }

    void main() {
      vUv = uv;
      
      // Base movement: particles start scattered and slowly migrate
      vec3 noise = curlNoise(position * 0.3 + time * 0.4);
      vec3 currentPos = mix(position, aTarget, uScroll);
      currentPos += noise * 1.2 * uScroll; 
      
      // Node Magnetism Field (Awwwards organic clustering)
      float pullRadius = 3.5;    // The range of the magnetic field
      float sphereRadius = 0.85; // The size of the invisible material node
      
      for(int i = 0; i < 7; i++) {
         vec3 nPos = uNodes[i];
         // Soft pulsation of the nodes
         nPos.y += sin(time * 1.5 + float(i) * 2.1) * 0.2;
         nPos.x += cos(time * 1.2 + float(i) * 1.7) * 0.1;
         
         vec3 dir = nPos - currentPos;
         float dist = length(dir);
         
         if(dist < pullRadius) {
             float strength = 1.0 - (dist / pullRadius);
             strength = smoothstep(0.0, 1.0, strength);
             
             // Wrap around the invisible material sphere
             vec3 surfacePos = nPos - normalize(dir) * sphereRadius;
             // Magnetic snap
             currentPos = mix(currentPos, surfacePos, strength * 0.9);
         }
      }

      // Camera orbital rotation
      float angle = time * 0.05 * (1.0 - uScroll);
      mat2 rot = mat2(cos(angle), -sin(angle), sin(angle), cos(angle));
      currentPos.xz = rot * currentPos.xz;

      vec4 mvPosition = modelViewMatrix * vec4(currentPos, 1.0);
      gl_Position = projectionMatrix * mvPosition;
      
      // Particles get bigger when closer to camera
      gl_PointSize = (16.0 / -mvPosition.z) * uScroll;
      vDistance = currentPos.z;
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

function GPUPoints({ count = 8000, color = "#00ff41" }) {
	const materialRef = useRef<THREE.ShaderMaterial>(null);

	const geometry = useMemo(() => {
		const pos = new Float32Array(count * 3);
		const tgt = new Float32Array(count * 3);

		for (let i = 0; i < count; i++) {
			const i3 = i * 3;
			
			// Initial chaotic position outside bounds (Cosmic dust)
			pos[i3] = (Math.random() - 0.5) * 50;
			pos[i3 + 1] = (Math.random() - 0.5) * 50;
			pos[i3 + 2] = (Math.random() - 0.5) * 30 - 10;

			// Base scattered target (before magnetism pulls them into neat spheres)
			tgt[i3] = (Math.random() - 0.5) * 15;
			tgt[i3 + 1] = (Math.random() - 0.5) * 15;
			tgt[i3 + 2] = (Math.random() - 0.5) * 8 - 2;
		}
		
		const geo = new THREE.BufferGeometry();
		geo.setAttribute('position', new THREE.BufferAttribute(pos, 3));
		geo.setAttribute('aTarget', new THREE.BufferAttribute(tgt, 3));
		return geo;
	}, [count]);

	const uniforms = useMemo(() => ({
		time: { value: 0 },
		uScroll: { value: 0 },
		uColor: { value: new THREE.Color(color) },
		uNodes: { value: nodeCenters }
	}), [color]);

	useFrame((state) => {
		if (materialRef.current) {
			const t = state.clock.elapsedTime;
			materialRef.current.uniforms.time.value = t;
			const canvasEl = state.gl.domElement;
			const rect = canvasEl.getBoundingClientRect();
			let progress = 1.0 - (rect.top / window.innerHeight);
			progress = Math.max(0, Math.min(1, progress * 1.5)); 
			
			materialRef.current.uniforms.uScroll.value += (progress - materialRef.current.uniforms.uScroll.value) * 0.05;
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
		const curves = [];
		const wasi = nodeCenters[3]; // Main central WASI defense node

		// Core logic: all connections go to WASI
		for (let i = 0; i < 7; i++) {
			if (i === 3) continue;
			const node = nodeCenters[i];
			const cp = new THREE.Vector3().addVectors(node, wasi).multiplyScalar(0.5);
			cp.z += 1.5 + Math.random() * 1.0; 
			curves.push(new THREE.QuadraticBezierCurve3(node, cp, wasi).getPoints(40));
		}

		// Some inter-node "rebuilding" routes that skip WASI
		const extra = [ [0, 2], [1, 2], [4, 5], [6, 5] ];
		extra.forEach(([from, to]) => {
			const n1 = nodeCenters[from];
			const n2 = nodeCenters[to];
			const cp = new THREE.Vector3().addVectors(n1, n2).multiplyScalar(0.5);
			cp.z -= 1.5;
			curves.push(new THREE.QuadraticBezierCurve3(n1, cp, n2).getPoints(30));
		});

		return curves;
	}, []);

	const impulsesRef = useRef<THREE.Group>(null);
	const glassRef = useRef<THREE.Group>(null);
	
	useFrame((state) => {
		const t = state.clock.elapsedTime;
		
		if (impulsesRef.current && glassRef.current) {
			impulsesRef.current.children.forEach((child: any, i: number) => {
				if (child.material && child.material.dashOffset !== undefined) {
					child.material.dashOffset -= 0.012; // Energy moving
					
					// Rebuilding & Disappearing Logic
					const lifeCycle = Math.sin(t * 0.8 + i * 1.3);
					if (lifeCycle < -0.6) {
						// Connection dropping/rebuilding
						child.material.opacity = 0.0;
						if (glassRef.current!.children[i]) {
							(glassRef.current!.children[i] as any).material.opacity = 0.0;
						}
					} else {
						// Connection exists
						const disruption = Math.sin(t * 5 + i * 2) * Math.sin(t * 1.2);
						child.material.opacity = disruption > 0 ? 0.9 : 0.2;
						if (glassRef.current!.children[i]) {
							(glassRef.current!.children[i] as any).material.opacity = 0.04;
						}
					}
				}
			});
		}
	});

	return (
		<group>
			{/* Glass Sheath (Visible only when connection is active) */}
			<group ref={glassRef}>
				{paths.map((pts, i) => (
					<Line 
						key={`glass-${i}`} 
						points={pts} 
						color={color} 
						lineWidth={4.0} 
						transparent 
						opacity={0.04} 
					/>
				))}
			</group>
			
			{/* Core Impulse */}
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

const PulsatingNode = ({ index, theme, children }: { index: number, theme: string, children: React.ReactNode }) => {
	const groupRef = useRef<THREE.Group>(null);
	const initialPos = nodeCenters[index];
	const textColor = theme === "dark" ? "#D1FFD7" : "#111"; 
	
	useFrame((state) => {
		if (groupRef.current) {
			const t = state.clock.elapsedTime;
			const yOffset = Math.sin(t * 1.5 + index * 2.1) * 0.2;
			const xOffset = Math.cos(t * 1.2 + index * 1.7) * 0.1;
			groupRef.current.position.set(
				initialPos.x + xOffset,
				initialPos.y + yOffset,
				initialPos.z
			);
		}
	});
	
	return (
		<group ref={groupRef} position={initialPos}>
			<Html center zIndexRange={[100, 0]}>
				<div style={{
					color: textColor,
					fontFamily: "monospace",
					fontSize: index === 3 ? "15px" : "12px", 
					fontWeight: "bold",
					textTransform: "uppercase",
					letterSpacing: "0.15em",
					whiteSpace: "nowrap",
					textShadow: theme === "dark" ? "0 0 15px rgba(0,255,65,0.8)" : "none",
					pointerEvents: "none",
				}}>
					{children}
				</div>
			</Html>
		</group>
	);
};

export function ParticleBackground({ theme }: { theme: "light" | "dark" }) {
	const color = theme === "dark" ? "#00ff41" : "#000000";

	return (
		<div style={{ position: "absolute", top: "10%", left: "50%", transform: "translateX(-50%)", width: "100%", height: "80%", zIndex: 0, pointerEvents: "none", opacity: theme === "dark" ? 0.9 : 0.3 }}>
			<Canvas camera={{ position: [0, 0, 14] }} gl={{ antialias: true, alpha: true }}>
				<ambientLight intensity={0.5} />
				<GPUPoints color={color} count={8000} />
				<Splines color={color} />
				
				{/* Nodes Overlay */}
				{nodeCenters.map((_, i) => (
					<PulsatingNode key={i} index={i} theme={theme}>
						{labels[i]}
					</PulsatingNode>
				))}
			</Canvas>
		</div>
	);
}
