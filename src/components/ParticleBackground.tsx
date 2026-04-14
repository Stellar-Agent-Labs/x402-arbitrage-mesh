"use client";

import { useMemo, useRef } from 'react';
import { Canvas, useFrame } from '@react-three/fiber';
import { Line } from '@react-three/drei';
import * as THREE from 'three';

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

function GPUPoints({ count = 8000, color = "#00ff41" }) {
	const materialRef = useRef<THREE.ShaderMaterial>(null);

	const geometry = useMemo(() => {
		const pos = new Float32Array(count * 3);
		const tgt = new Float32Array(count * 3);

		const nodeCenters = [
			new THREE.Vector3(-4.5, 4.5, 0), // HTTP 
			new THREE.Vector3(4.5, 4.5, 0),  // Farcaster 
			new THREE.Vector3(0, 2.25, 0),   // Gateway 
			new THREE.Vector3(0, -0.75, 1),  // WASM Sandbox Z-elevated
			new THREE.Vector3(-4.5, -3.75, 0),// Tier 1 
			new THREE.Vector3(0, -3.75, 0),   // Tier 2 
			new THREE.Vector3(4.5, -3.75, 0), // Tier 3 
		];

		for (let i = 0; i < count; i++) {
			const r = 5 + Math.random() * 8;
			const theta = Math.random() * 2 * Math.PI;
			const phi = Math.acos((Math.random() * 2) - 1);
			pos[i * 3] = r * Math.sin(phi) * Math.cos(theta);
			pos[i * 3 + 1] = r * Math.sin(phi) * Math.sin(theta);
			pos[i * 3 + 2] = r * Math.cos(phi);

			const center = nodeCenters[Math.floor(Math.random() * nodeCenters.length)];
			const jr = Math.random() * 1.5;
			const jt = Math.random() * 2 * Math.PI;
			const jp = Math.acos((Math.random() * 2) - 1);
			tgt[i * 3] = center.x + jr * Math.sin(jp) * Math.cos(jt);
			tgt[i * 3 + 1] = center.y + jr * Math.sin(jp) * Math.sin(jt);
			tgt[i * 3 + 2] = center.z + jr * Math.cos(jp);
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
			materialRef.current.uniforms.time.value = state.clock.elapsedTime;
			// Use real DOM canvas bounds to precisely trigger magnetism
			const canvasEl = state.gl.domElement;
			const rect = canvasEl.getBoundingClientRect();
			let progress = 1.0 - (rect.top / window.innerHeight);
			progress = Math.max(0, Math.min(1, progress * 1.5)); // 1.5 multiplier makes it magnetize faster when in view
			
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

	const groupRef = useRef<THREE.Group>(null);
	
	useFrame(() => {
		if (groupRef.current) {
			groupRef.current.children.forEach((child: any) => {
				if (child.material && child.material.dashOffset !== undefined) {
					child.material.dashOffset -= 0.008;
				}
			});
		}
	});

	return (
		<group ref={groupRef}>
			{paths.map((pts, i) => (
				<Line 
					key={i} 
					points={pts} 
					color={color} 
					lineWidth={2.5} 
					dashed 
					dashScale={15} 
					dashSize={0.8} 
					dashOffset={0} 
					transparent 
					opacity={0.8} 
				/>
			))}
		</group>
	);
}

export function ParticleBackground({ theme }: { theme: "light" | "dark" }) {
	const color = theme === "dark" ? "#00ff41" : "#000000";

	return (
		<div style={{ position: "absolute", top: "10%", left: "50%", transform: "translateX(-50%)", width: "100%", height: "80%", zIndex: 0, pointerEvents: "none", opacity: theme === "dark" ? 0.9 : 0.3 }}>
			<Canvas camera={{ position: [0, 0, 14] }} gl={{ antialias: true, alpha: true }}>
				<ambientLight intensity={0.5} />
				<GPUPoints color={color} count={12000} />
				<Splines color={color} />
			</Canvas>
		</div>
	);
}
