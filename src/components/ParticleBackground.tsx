"use client";

import { useMemo, useRef } from 'react';
import { Canvas, useFrame } from '@react-three/fiber';
import { Text, Environment } from '@react-three/drei';
import * as THREE from 'three';

// 1. Стеклянные кольца с дисперсией
function GlassRings({ theme }: { theme: "light" | "dark" }) {
	const ringGroups = useRef<THREE.Group[]>([]);
	const materialColor = theme === "dark" ? "#ffffff" : "#cfcfcf";

	// Конфигурация орбит: радиус, толщина самой трубки, изначальный наклон (Euler), скорость вращения
	const ringsData = useMemo(() => [
		{ radius: 3.5, tube: 0.12, rotation: [Math.PI / 3, Math.PI / 4, 0], speed: 0.15, label: "TIER 1: LOCAL LLM" },
		{ radius: 5.0, tube: 0.08, rotation: [-Math.PI / 6, Math.PI / 3, Math.PI / 6], speed: -0.12, label: "FARCASTER GATE" },
		{ radius: 6.5, tube: 0.18, rotation: [0, -Math.PI / 4, Math.PI / 8], speed: 0.08, label: "SOVEREIGN GATEWAY" },
		{ radius: 8.0, tube: 0.05, rotation: [Math.PI / 2, 0, -Math.PI / 6], speed: -0.05, label: "HTTP AGENT" },
	], []);

	useFrame((state, delta) => {
		ringGroups.current.forEach((group, i) => {
			if (group) {
				// Вращаем саму группу (содержащую и кольцо, и текст)
				group.rotation.y += ringsData[i].speed * delta;
				group.rotation.z += ringsData[i].speed * delta * 0.4;
			}
		});
	});

	// Абсолютный Awwwards-материал стекла (преломление)
	const glassMaterial = useMemo(() => new THREE.MeshPhysicalMaterial({
		color: materialColor,
		metalness: 0.2,
		roughness: 0.0,
		transmission: 1.0, // Полная прозрачность как стекло
		ior: 1.6, // Index of Refraction (Искажает частицы внутри)
		thickness: 2.0, // Толщина преломляющего объема
		clearcoat: 1.0,
		clearcoatRoughness: 0.1,
		transparent: true,
		opacity: 1.0,
		side: THREE.DoubleSide
	}), [materialColor]);

	return (
		<>
			{ringsData.map((data, i) => (
				<group 
					key={`ring-${i}`} 
					ref={(el) => { if (el) ringGroups.current[i] = el; }} 
					rotation={new THREE.Euler(...data.rotation)}
				>
					{/* Само стеклянное кольцо */}
					<mesh material={glassMaterial}>
						<torusGeometry args={[data.radius, data.tube, 64, 128]} />
					</mesh>
					
					{/* Бегущий 3D Текст: Прикреплен к внешней кромке кольца */}
					<Text
						position={[data.radius + data.tube + 0.3, 0, 0]}
						rotation={[0, Math.PI / 2, 0]} // Лицом наружу
						fontSize={0.35}
						color={theme === "dark" ? "#00ff41" : "#111111"}
						font="https://fonts.gstatic.com/s/roboto/v30/KFOmCnqEu92Fr1Mu4mxK.woff" // Дефолтный шрифт
						anchorX="center"
						anchorY="middle"
						fillOpacity={0.9}
					>
						{data.label}
					</Text>
					{/* Симметричный лейбл на другой стороне орбиты */}
					<Text
						position={[-(data.radius + data.tube + 0.3), 0, 0]}
						rotation={[0, -Math.PI / 2, 0]}
						fontSize={0.35}
						color={theme === "dark" ? "#00ff41" : "#111111"}
						font="https://fonts.gstatic.com/s/roboto/v30/KFOmCnqEu92Fr1Mu4mxK.woff"
						anchorX="center"
						anchorY="middle"
						fillOpacity={0.9}
					>
						{data.label}
					</Text>
				</group>
			))}
		</>
	);
}

// 2. Центральное Ядро WASI (Прохладные частицы)
function CoreParticles() {
	const pointsRef = useRef<THREE.Points>(null);
	const count = 6000;
	
	// Шейдер для легкой пульсации ("дыхания") сферы
	const vertexShader = `
		uniform float time;
		varying vec3 vColor;
		void main() {
			vec3 pos = position;
			// Мягкая пульсация всей массы частиц
			float pulse = sin(time * 2.5 + length(pos)*5.0) * 0.08;
			pos += normalize(pos) * pulse;
			
			// Плавное вращение самого ядра
			float angle = time * 0.3;
			mat2 rot = mat2(cos(angle), -sin(angle), sin(angle), cos(angle));
			pos.xz = rot * pos.xz;
			
			vec4 mvPosition = modelViewMatrix * vec4(pos, 1.0);
			gl_Position = projectionMatrix * mvPosition;
			
			// Размеры частиц настраиваются тут
			gl_PointSize = (12.0 / -mvPosition.z);
			
			// Прохладная палитра (Cyan -> Mint Green)
			float dist = length(position) / 2.0;
			vColor = mix(vec3(0.0, 0.8, 1.0), vec3(0.0, 1.0, 0.5), dist);
		}
	`;

	const fragmentShader = `
		varying vec3 vColor;
		void main() {
			// Рисуем кружок с мягким краем
			float dist = length(gl_PointCoord - vec2(0.5));
			if (dist > 0.5) discard;
			float alpha = smoothstep(0.5, 0.1, dist);
			gl_FragColor = vec4(vColor, alpha * 0.9);
		}
	`;

	const geometry = useMemo(() => {
		const pos = new Float32Array(count * 3);
		// Заполняем плотную сферу полярными координатами
		for (let i = 0; i < count; i++) {
			const u = Math.random();
			const v = Math.random();
			const theta = u * 2.0 * Math.PI;
			const phi = Math.acos(2.0 * v - 1.0);
			const r = Math.cbrt(Math.random()) * 2.0; // Радиус ядра WASI = 2.0
			
			pos[i * 3]     = r * Math.sin(phi) * Math.cos(theta);
			pos[i * 3 + 1] = r * Math.sin(phi) * Math.sin(theta);
			pos[i * 3 + 2] = r * Math.cos(phi);
		}
		const geo = new THREE.BufferGeometry();
		geo.setAttribute('position', new THREE.BufferAttribute(pos, 3));
		return geo;
	}, []);
	
	const uniforms = useMemo(() => ({
		time: { value: 0 }
	}), []);

	useFrame((state) => {
		if (pointsRef.current) {
			(pointsRef.current.material as THREE.ShaderMaterial).uniforms.time.value = state.clock.elapsedTime;
		}
	});

	return (
		<group>
			{/* Само ядро */}
			<points ref={pointsRef} geometry={geometry}>
				<shaderMaterial
					vertexShader={vertexShader}
					fragmentShader={fragmentShader}
					uniforms={uniforms}
					transparent={true}
					blending={THREE.AdditiveBlending}
					depthWrite={false}
				/>
			</points>
			{/* Текст поверх Ядра WASI */}
			<Text
				position={[0, 0, 2.5]} // Выдвинут немного вперед поверх ядра
				fontSize={0.6}
				color="#00ffff"
				font="https://fonts.gstatic.com/s/roboto/v30/KFOmCnqEu92Fr1Mu4mxK.woff"
				anchorX="center"
				anchorY="middle"
			>
				WASM SANDBOX (CORE)
			</Text>
		</group>
	);
}

export function ParticleBackground({ theme }: { theme: "light" | "dark" }) {
	return (
		<div style={{ position: "absolute", top: "10%", left: "50%", transform: "translateX(-50%)", width: "100%", height: "80%", zIndex: 0, pointerEvents: "none", opacity: theme === "dark" ? 0.95 : 0.8 }}>
			<Canvas camera={{ position: [0, 0, 15], fov: 45 }} gl={{ antialias: true, alpha: true }}>
				{/* 
					HDRI окружение КРИТИЧНО ВАЖНО для стекла (MeshPhysicalMaterial). 
					Оно дает отражения среды и реалистичное световое преломление. 
				*/}
				<Environment preset="city" />
				
				{/* Источники света для подсветки граней колец */}
				<ambientLight intensity={0.5} />
				<directionalLight position={[10, 10, 5]} intensity={2.0} color="#00ffff" />
				<directionalLight position={[-10, -10, -5]} intensity={1.0} color="#00ff41" />

				{/* 1. Центр: Модуль защиты WASI */}
				<CoreParticles />
				
				{/* 2. Орбиты: Стеклянные кольца с метками нод */}
				<GlassRings theme={theme} />
			</Canvas>
		</div>
	);
}
