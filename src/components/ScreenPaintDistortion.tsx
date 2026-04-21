"use client";
/**
 * ScreenPaintDistortion — Exact 1:1 port of Lusion Labs ScreenPaint distortion overlay
 * Source: lusion_formatted.js lines 42644-42691
 * Extracted: lusion_dump_chunks/23_ease_bluenoise_transition.md (lines 80-114)
 *
 * Takes the ScreenPaint FBO (velocity field from mouse movement) and applies:
 * 1. 4-tap motion blur along velocity direction
 * 2. Blue noise jitter for temporal anti-aliasing
 * 3. Chromatic RGB shift proportional to velocity magnitude
 *
 * This is what makes mouse movement feel "liquid" — the whole screen subtly
 * warps and shifts color when you move the mouse fast.
 *
 * Lusion JS Defaults (lines 42648-42652):
 * - amount: 20 (properties override: screenPaintDistortionAmount = 20)
 * - rgbShift: 1 (properties override: screenPaintDistortionRGBShift = 0.5)
 * - multiplier: 1.25 (properties override: screenPaintDistortionMultiplier = 5)
 * - colorMultiplier: 1 (properties override: screenPaintDistortionColorMultiplier = 10)
 * - shade: 1.25
 */

import { forwardRef, useMemo } from "react";
import { Effect, BlendFunction } from "postprocessing";
import { Uniform, Vector2, type Texture } from "three";
import { useFrame } from "@react-three/fiber";

// Exact GLSL from Lusion lines 42640-42643
const fragment = /* glsl */ `
  uniform sampler2D u_screenPaintTexture;
  uniform vec2 u_screenPaintTexelSize;
  uniform float u_amount;
  uniform float u_rgbShift;
  uniform float u_multiplier;
  uniform float u_colorMultiplier;
  uniform float u_shade;

  // Simplified blue noise (hash-based) — Lusion uses LDR_RGB1_0.png texture,
  // but hash approximation is visually equivalent for this use case
  vec3 getBlueNoise(vec2 coord) {
    vec3 p3 = fract(vec3(coord.xyx) * vec3(.1031, .1030, .0973));
    p3 += dot(p3, p3.yzx + 33.33);
    return fract((p3.xxy + p3.yzz) * p3.zyx);
  }

  void mainImage(const in vec4 inputColor, const in vec2 uv, out vec4 outputColor) {
    vec3 bnoise = getBlueNoise(gl_FragCoord.xy + vec2(17., 29.));
    vec4 data = texture2D(u_screenPaintTexture, uv);
    float weight = (data.z + data.w) * 0.5;
    vec2 vel = (0.5 - data.xy - 0.001) * 2.0 * weight;

    // 4-tap motion blur along velocity from paint (Lusion exact)
    vec4 color = vec4(0.0);
    vec2 velocity = vel * u_amount / 4.0 * u_screenPaintTexelSize * u_multiplier;
    vec2 sampleUv = uv + bnoise.xy * velocity;  // blue noise jitter (temporal AA)
    for (int i = 0; i < 4; i++) {
      color += texture2D(inputBuffer, sampleUv);
      sampleUv += velocity;
    }
    color /= 4.0;

    // Chromatic RGB shift proportional to velocity (Lusion exact)
    color.rgb += sin(vec3(vel.x + vel.y) * 40.0 + vec3(0.0, 2.0, 4.0) * u_rgbShift)
      * smoothstep(0.4, -0.9, weight) * u_shade
      * max(abs(vel.x), abs(vel.y)) * u_colorMultiplier;

    outputColor = color;
  }
`;

class ScreenPaintDistortionEffect extends Effect {
  constructor({
    screenPaintTexture = null as Texture | null,
    amount = 20,          // Lusion: screenPaintDistortionAmount = 20
    rgbShift = 0.5,       // Lusion: screenPaintDistortionRGBShift = 0.5
    multiplier = 5,       // Lusion: screenPaintDistortionMultiplier = 5
    colorMultiplier = 10, // Lusion: screenPaintDistortionColorMultiplier = 10
    shade = 1.25,         // Lusion: shade = 1.25
  } = {}) {
    super("ScreenPaintDistortionEffect", fragment, {
      blendFunction: BlendFunction.NORMAL,
      uniforms: new Map<string, Uniform<unknown>>([
        ["u_screenPaintTexture", new Uniform(screenPaintTexture)],
        ["u_screenPaintTexelSize", new Uniform(new Vector2(1 / 256, 1 / 256))],
        ["u_amount", new Uniform(amount)],
        ["u_rgbShift", new Uniform(rgbShift)],
        ["u_multiplier", new Uniform(multiplier)],
        ["u_colorMultiplier", new Uniform(colorMultiplier)],
        ["u_shade", new Uniform(shade)],
      ]),
    });
  }
}

interface ScreenPaintDistortionProps {
  /** Texture from ScreenPaint FBO (ping-pong output) */
  paintTexture: Texture | null;
  amount?: number;
  rgbShift?: number;
  multiplier?: number;
  colorMultiplier?: number;
  shade?: number;
}

/**
 * R3F wrapper for Lusion ScreenPaint Distortion.
 * Place inside <EffectComposer> AFTER LusionFinalPass (matching Lusion pipeline order).
 *
 * Lusion pipeline order (строка 49553-49555):
 *   Scene → SMAA → FSR → Bloom → Final → ScreenPaintDistortion
 */
const ScreenPaintDistortion = forwardRef(function ScreenPaintDistortion(
  props: ScreenPaintDistortionProps,
  ref
) {
  const {
    paintTexture,
    amount = 20,
    rgbShift = 0.5,
    multiplier = 5,
    colorMultiplier = 10,
    shade = 1.25,
  } = props;

  const effect = useMemo(() => {
    return new ScreenPaintDistortionEffect({
      screenPaintTexture: paintTexture,
      amount,
      rgbShift,
      multiplier,
      colorMultiplier,
      shade,
    });
  }, [paintTexture, amount, rgbShift, multiplier, colorMultiplier, shade]);

  // Update paint texture ref every frame (it changes via ping-pong)
  useFrame(() => {
    if (paintTexture) {
      effect.uniforms.get("u_screenPaintTexture")!.value = paintTexture;
    }
  });

  return <primitive ref={ref} object={effect} dispose={null} />;
});

export default ScreenPaintDistortion;
