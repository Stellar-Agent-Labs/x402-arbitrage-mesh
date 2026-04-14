"use client";
import { Icosahedron, MeshTransmissionMaterial } from "@react-three/drei";
import { useFrame } from "@react-three/fiber";
import { useRef } from "react";
import type * as THREE from "three";

export default function RefractiveCore() {
	const meshRef = useRef<THREE.Mesh>(null);

	useFrame((state) => {
		if (meshRef.current) {
			meshRef.current.rotation.x = state.clock.elapsedTime * 0.15;
			meshRef.current.rotation.y = state.clock.elapsedTime * 0.25;
			// Легкое левитирование (Lusion effect)
			meshRef.current.position.y = Math.sin(state.clock.elapsedTime) * 0.3;
		}
	});

	return (
		<Icosahedron ref={meshRef} args={[3.5, 4]} position={[0, 0, 0]}>
			<MeshTransmissionMaterial
				backside
				samples={5}
				thickness={3.5}
				chromaticAberration={2}
				anisotropy={0.5}
				distortion={0.8}
				distortionScale={0.5}
				temporalDistortion={0.3}
				transmission={1.0}
				roughness={0.05}
				ior={1.6}
				color="#00ff41"
				resolution={1024}
			/>
		</Icosahedron>
	);
}
