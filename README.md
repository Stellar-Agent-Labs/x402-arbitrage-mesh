<div align="center">
  <img src="https://upload.wikimedia.org/wikipedia/commons/thumb/1/14/Stellar_symbol_black.svg/2048px-Stellar_symbol_black.svg.png" width="100" alt="Stellar Logo" />
  <h1 align="center">The x402 Arbitrage Mesh</h1>
  <p align="center">
    <strong>Autonomous Inter-Agent Payment Routing Protocol on Soroban</strong>
  </p>
  <p align="center">
    <i>Submission for DoraHacks: Stellar Hacks 2026</i>
  </p>
</div>

<br />

## 🌐 The Agentic Economy is Broken. We Fixed It.

The current AI ecosystem (OpenClaw, ElizaOS, Virtuals) relies on isolated agents. If an agent is overwhelmed, it drops tasks. If a task is too complex, the client suffers. 

**The x402 Arbitrage Mesh** is the world's first decentralized Load Balancer and Payment Router for AI Agents, built natively on the **Stellar Network** using the **L402 (Payment Required)** protocol.

We don't just enable Agent-to-Agent payments. We enable **Agent-to-Agent Delegation**.

## 🔥 Core Architecture (The Autonomous Dispatcher)

Our Gateway acts as an intelligent, transparent Hub for the Agentic Social Network:
1. **The Greed Classifier:** An external AI task is submitted via HTTP `POST /api/hire` with a Soroban USDC L402 signature.
2. **Enterprise Execution:** If the micro-bounty is high (e.g., >= 5.00 USDC), the Gateway assigns it to an isolated, high-compute Sovereign Node.
3. **P2P Uber-Arbitrage:** If the bounty is low, the Gateway instantly delegates the task to an idle agent on the P2P network, pays them via a Soroban smart contract, and extracts a 0% P2P routing fee. 

*Result: Infinite scalability. Zero dropped tasks. Instant USDC settlement.*

---

## 🚀 Quick Start (For Judges)

### 1. Installation
Clone the repository and install dependencies:
```bash
git clone https://github.com/x402-mesh/x402-arbitrage-gateway.git
cd x402-arbitrage-gateway
npm install
```

### 2. Run the Gateway (UI + Edge Router)
```bash
npm run dev
```
Navigate to `http://localhost:3000`. You will see the **Glassmorphism Terminal Feed** visualizing real-time Inter-Agent communication and Soroban L402 payment routing.

### 3. Run the P2P Network Mock (External Agent)
To demonstrate the "Uber Arbitrage" delegation in real-time, spin up our OpenClaw mock-agent in a separate terminal:
```bash
node dummy_external_bot.js
```
Now, submit a task < $5.00 USDC in the UI. Watch the Gateway instantly route the task and payment to the external agent on port 3001.

---

## 🔒 Security & OPSEC
The Mesh is secured by an internal Extism WASM Quarantine layer. Any code or JSON payload returned by an untrusted external mercenary bot is parsed and sanitized in a zero-trust WASI 0.2 sandbox before being returned to the original client.

## 📜 License
MIT License. Open for the Swarm. Built for Stellar.
