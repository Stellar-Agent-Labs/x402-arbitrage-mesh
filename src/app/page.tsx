"use client";
import React from "react";
import { motion } from "framer-motion";
import LiquidGlassShader from "@/components/LiquidGlassShader";
import HollywoodTelemetry from "@/components/HollywoodTelemetry";

export default function Page() {
  return (
    <main style={{ minHeight: "100vh", position: "relative", overflow: "hidden" }}>
      <LiquidGlassShader />
      
      <motion.div 
        initial={{ opacity: 0, y: 50 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 1.5, ease: "easeOut" }}
        style={{ position: "relative", zIndex: 10, display: "flex", flexDirection: "column", alignItems: "center", justifyItems: "center", paddingTop: "15vh", height: "100vh", padding: "2rem" }}
      >
        <motion.h1 
           initial={{ scale: 0.9, opacity: 0 }}
           animate={{ scale: 1, opacity: 1 }}
           transition={{ delay: 0.5, duration: 2, ease: "anticipate" }}
           style={{ color: "#fff", fontSize: "5rem", letterSpacing: "0.15em", fontFamily: "monospace", textShadow: "0 0 40px rgba(0, 255, 65, 0.9)", margin: 0, textAlign: 'center' }}
        >
          SOVEREIGN GATEWAY
        </motion.h1>
        <motion.p 
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ delay: 1.5, duration: 1 }}
          style={{ color: "#00ff41", fontFamily: "monospace", fontSize: "1.2rem", marginTop: "1rem", letterSpacing: "0.6em", textTransform: 'uppercase' }}
        >
          x402 protocol / Lusion Matrix
        </motion.p>
        
        <motion.div 
          whileHover={{ scale: 1.05, boxShadow: "0 0 40px rgba(0,255,65,0.7)" }}
          whileTap={{ scale: 0.95 }}
          style={{ marginTop: "4rem", padding: "2rem 4rem", border: "1px solid rgba(0, 255, 65, 0.5)", borderRadius: "4px", background: "rgba(0,15,0,0.4)", backdropFilter: "blur(15px)", cursor: "pointer", position: "relative", overflow: "hidden" }}
        >
          <div style={{ position: "absolute", top: 0, left: 0, right: 0, height: "1px", background: "linear-gradient(90deg, transparent, #00ff41, transparent)" }} />
          <div style={{ position: "absolute", bottom: 0, left: 0, right: 0, height: "1px", background: "linear-gradient(90deg, transparent, #00ff41, transparent)" }} />
          <p style={{ color: "#fff", fontFamily: "monospace", textAlign: "center", margin: 0, fontWeight: "bold", letterSpacing: "0.2em", fontSize: "1.1rem" }}>
            [ INITIATE L402 HANDSHAKE ]
          </p>
        </motion.div>
      </motion.div>

      <motion.div
        initial={{ opacity: 0, x: -50 }}
        animate={{ opacity: 1, x: 0 }}
        transition={{ delay: 2, duration: 1 }}
      >
        <HollywoodTelemetry />
      </motion.div>
    </main>
  );
}
