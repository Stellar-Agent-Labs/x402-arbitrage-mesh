import React from "react";
import LiquidGlassShader from "@/components/LiquidGlassShader";
import HollywoodTelemetry from "@/components/HollywoodTelemetry";

export default function Page() {
  return (
    <main style={{ minHeight: "100vh", position: "relative", overflow: "hidden" }}>
      <LiquidGlassShader />
      
      <div style={{ position: "relative", zIndex: 10, display: "flex", flexDirection: "column", alignItems: "center", justifyContent: "center", height: "100vh", padding: "2rem" }}>
        <h1 style={{ color: "#fff", fontSize: "4rem", fontFamily: "monospace", textShadow: "0 0 20px #00ff41" }}>
          TRIARCHY GATEWAY
        </h1>
        <p style={{ color: "#00ff41", fontFamily: "monospace", fontSize: "1.5rem", marginTop: "1rem" }}>
          STATUS: SOVEREIGN
        </p>
        
        <div style={{ marginTop: "3rem", padding: "2rem", border: "1px solid rgba(0, 255, 65, 0.5)", borderRadius: "8px", background: "rgba(0,0,0,0.5)", backdropFilter: "blur(5px)" }}>
          <p style={{ color: "#fff", fontFamily: "monospace", textAlign: "center" }}>
            Awaiting x402 Soroban L402 Payment Extism Validation...
          </p>
        </div>
      </div>

      <HollywoodTelemetry />
    </main>
  );
}
