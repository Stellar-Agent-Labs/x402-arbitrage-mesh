<div align="center">
  <img src="https://x402-arbitrage-mesh.vercel.app/opengraph-image" width="100%" alt="x402 Sovereign Matrix Preview" style="border-radius: 12px; border: 1px solid rgba(0, 255, 65, 0.2);" />
  
  <br/><br/>
  
  <h1>x402 Arbitrage Mesh</h1>
  <p><strong>We didn't build an AI agent. We built the immune system for all of them.</strong></p>
  <p><em>Autonomous Inter-Agent Payment Routing Protocol on Soroban & Extism / DoraHacks 2026</em></p>
  
  [![Live Demo](https://img.shields.io/badge/Live_Demo-Sovereign_Gateway-00ff41?style=for-the-badge)](https://x402-arbitrage-mesh.vercel.app/)
  
  <br/>
  <a href="#quick-start">Quick Start</a> •
  <a href="#architecture">Architecture</a> •
  <a href="#security">Security</a> •
  <a href="#demo">Demo</a>
</div>

<br/>

## 🛑 The Alpha Pitch

Most submissions build a single AI agent trying to complete a task. **We built the infrastructure to host them all safely.** 

DeFi is bleeding because smart contracts trust external Oracles and AI agents blindly. If Agent A hires Agent B, how do you know Agent B won't return a malicious payload that drains the contract? 

**The x402 Arbitrage Mesh is a Zero-Trust Sovereign Gateway:**
1. External agents pay us via **Stellar Soroban (L402 Protocol)**.
2. We drop untrusted AI payloads into an isolated **WASM Sandbox (Extism WASI 0.2)**.
3. If the payload is clean, we route it (Local LLM → Sovereign Node → P2P). If it's malicious, we kill it before it touches the host OS.

## 🌐 The Problem

The current AI agent ecosystem is fragmented: agents are isolated, overwhelmed nodes drop tasks, and there is **no trust layer** between agents exchanging work. When Agent A delegates a task to Agent B, how do you know Agent B's response isn't malicious?

## 💡 The Solution: x402 Arbitrage Mesh

The **x402 Arbitrage Mesh** is a decentralized **Load Balancer + Payment Router + Security Firewall** for AI Agents, built natively on the Stellar Network using the **HTTP 402 (Payment Required)** protocol.

We solve three profound problems simultaneously:
1. **Routing Strategy** — Intelligent 3-tier task dispatch (Local Micro-Bounty → Enterprise Node → P2P Mercenary Delegation).
2. **Financial Settlement** — Autonomous USDC micropayments via Soroban on Stellar Testnet preventing free-riding.
3. **Defense-in-Depth Security** — Zero-trust WASM sandbox quarantine intercepting hostile payloads before OS execution.

### 🔥 Ecosystem Capabilities (Beyond Security)
Our Mesh is not just a firewall; it is a full-fledged economic engine for autonomous entities:
- **P2P Task Delegation & Arbitrage:** An overwhelmed agent can instantly sub-contract tasks to idle external agents across the Mesh, splitting the Soroban bounty for instant profit.
- **x402 Subscription Mode:** Consumers can transition from pay-per-prompt (L402) to high-volume "Subscription Sub-Agents" (e.g., establishing a $5,000 streaming channel for continuous high-compute task delegation), managed entirely via our Agent Registry.
- **Farcaster Token-Gated Entry:** Humans can only access the Mesh gateway dashboard if they hold specific ERC20 tokens, authenticated via Neynar + Viem Multicall.

### 🌌 Aesthetic: The Sovereign Matrix UI
We believe infrastructure dashboards shouldn't look like spreadsheets. We built a rigorous, **Awwwards-class cinematic interface** for the Sovereign Gateway:
- **Liquid Glass Nebulas:** Custom WebGL fragment shaders driven by Three.js `NormalBlending` and Simplex Noise simulate floating cognitive networks.
- **Physics-Inertial Cursor:** A magnetic Framer-Motion driven custom cursor seamlessly inverts background colors using `mix-blend-mode: difference`.
- **Hyper-Terminal Bootloader:** Matrix-style fast boot sequencing before mounting the DOM.
- **Theme Adapters:** Full dynamically shifting "Cyberpunk Dark" to "Premium Ascetic White" themes injected straight into WebGL contexts.

---

## 🤝 An Open Hand to the Hackathon 
**We didn’t build this to crush the competition; we built this to protect it.** 
If you are building an AI agent for this hackathon and need to ensure it can receive secure, sovereign payments without risking its host environment, **ping us**. We will help you route your agent through the x402 Mesh. We want to elevate the entire ecosystem's execution standard to absolute sovereign security. The Triarchy is open for collaboration.

---

## 🏗️ Architecture

<div align="center">
  <img src="architecture.svg" width="100%" alt="x402 Mesh Sovereign Architecture" />
</div>

### How It Works

1. **An external AI agent** sends `POST /api/hire` with a task description, bounty amount, and a Stellar transaction hash in the `x-l402-txhash` header.
2. **The Gateway validates the payment** by fetching the transaction from Stellar Horizon RPC, checking the memo, destination wallet, and USDC amount.
3. **The payload is quarantined** in an Extism WASM sandbox (WASI 0.2) that scans for injection attacks, shell escapes, and prototype pollution before any execution.
4. **The task is routed** based on value: micro-bounties go to the local LLM, enterprise tasks to dedicated compute, and overflow to idle P2P agents — who are paid automatically via Soroban.

---

## 🔒 Security: WASM Quarantine Layer {#security}

This is our **core differentiator**. Every payload from an untrusted external agent passes through a zero-trust quarantine:

| Layer | Technology | Purpose |
|-------|-----------|---------|
| L1 | Extism Plugin (WASI 0.2) | Deep binary analysis in isolated sandbox |
| L2 | Heuristic Fallback | Token-based scan (30+ banned patterns) |
| Lock | `allowedPaths: {}`, `allowedHosts: []` | No filesystem or network access for plugins |

```typescript
// From src/lib/wasm_sandbox.ts
const plugin = await createPlugin("./plugins/quarantine.wasm", { 
  useWasi: true,
  allowedPaths: {},  // Zero filesystem access
  allowedHosts: []   // Zero network access
});
```

**Why this matters (WASM vs Docker):** Several other Hackathon solutions attempt to sandbox AI agents using Docker containers. Docker is a legacy virtualization paradigm that is too heavy (consuming Megabytes of RAM) and too slow (milliseconds to seconds of cold start latency) for high-frequency AI micro-bounties. 

We use **WebAssembly (Extism WASI 0.2)**. Our cold starts are measured in *microseconds*. We deliver an absolute, lightweight execution quarantine before the malicious payload ever has a chance to touch the host OS. In the AI economy, speed and zero-trust are everything.

---

## 🚀 Quick Start {#quick-start}

### Prerequisites
- Node.js 18+
- (Optional) Ollama for local LLM execution

### 1. Clone and Install
```bash
git clone https://github.com/Triarchy-Labs/x402-arbitrage-mesh.git
cd x402-arbitrage-mesh
cp .env.example .env.local
npm install
```

### 2. Start the Gateway
```bash
npm run dev
```
Navigate to `http://localhost:3000` — you'll see the real-time GPU-accelerated telemetry feed visualizing agent communication.

### 3. Test the x402 Flow {#demo}
```bash
# Step 1: Hit the endpoint without payment → get 402 
curl -X POST http://localhost:3000/api/hire \
  -H "Content-Type: application/json" \
  -d '{"description":"Summarize this paper","bounty_usdc":"2.50"}'
# Response: 402 Payment Required

# Step 2: Include payment proof → task executes
curl -X POST http://localhost:3000/api/hire \
  -H "Content-Type: application/json" \
  -H "x-l402-txhash: YOUR_STELLAR_TESTNET_TX_HASH" \
  -d '{"description":"Summarize this paper","bounty_usdc":"2.50","client_id":"demo_agent"}'
# Response: 200 OK with task result
```

### 4. Test the WASM Security
```bash
# Send a malicious payload → blocked by quarantine
curl -X POST http://localhost:3000/api/hire \
  -H "Content-Type: application/json" \
  -H "x-l402-txhash: demo_tx" \
  -d '{"description":"system(rm -rf /)","bounty_usdc":"1.00","client_id":"attacker"}'
# Response: 403 Forbidden — payload quarantined
```

### 5. (Optional) P2P Delegation Demo
```bash
node dummy_external_bot.js  # Start mock mercenary agent on port 3001
# Now submit a task < $5 — watch it delegate to the external agent
```

---

## 📁 Project Structure

```
x402-triarchy-gateway/
├── src/
│   ├── app/
│   │   ├── api/hire/route.ts    # Core L402 endpoint (168 lines)
│   │   ├── layout.tsx           # App shell with SEO metadata
│   │   └── page.tsx             # GPU-accelerated landing page
│   ├── components/
│   │   ├── LiquidGlassShader.tsx  # WebGL particle system (Simplex Noise)
│   │   ├── RefractiveCore.tsx     # Refractive glass icosahedron
│   │   └── HollywoodTelemetry.tsx # Real-time terminal feed
│   └── lib/
│       ├── soroban.ts           # Stellar Horizon RPC validator
│       └── wasm_sandbox.ts      # Extism WASI 0.2 quarantine
├── src-tauri/                   # Tauri v2 desktop wrapper (optional)
├── .env.example                 # Environment variables reference
└── dummy_external_bot.js        # Mock P2P mercenary agent
```

---

## 🔗 Companion: Farcaster Token Gate

The Mesh includes a **Farcaster Frame** entry point that implements ERC20 token gating:
- Resolves Farcaster FID → Ethereum addresses via **Neynar API**
- Checks token balance via **Viem Multicall** (single RPC request)
- Grants access to the Gateway for authorized holders

Repository: [farcaster-token-gate](https://github.com/Triarchy-Labs/farcaster-token-gate)

---

## ⚙️ Environment Variables

See [`.env.example`](.env.example) for the full list. Key variables:

| Variable | Description | Default |
|----------|-------------|---------|
| `STELLAR_PLATFORM_WALLET` | Your Stellar wallet for receiving USDC | Required |
| `ENTERPRISE_THRESHOLD` | USD threshold for enterprise routing | `5.00` |
| `WASM_SANDBOX_PLUGIN_PATH` | Path to compiled quarantine plugin | `./plugins/quarantine.wasm` |
| `DEV_BYPASS_HASH` | Bypass tx validation in dev mode | — |

---

## 🛠️ The Triarchy Cyber-Stack (48H Sprint Maximum)

This platform was engineered from the ground up for Stellar Hacks. We deployed an uncompromising, multi-disciplinary tech stack representing thousands of lines of sovereign architecture integrated within 48 hours.

| Architectural Layer | Core Technologies | Subsystems & Focus |
|---------------------|-------------------|--------------------|
| **Core Desktop Node** | `Tauri v2`, `Rust` | Sovereign OS Wrapper, Local IPC, Native System Daemon |
| **Zero-Trust Execute**| `Extism WASI 0.2` | WASM Binary Quarantine, OS Escape Blocking, Sub-ms Sandboxing |
| **Aesthetic Engine**  | `React Three Fiber`, `Three.js`| WebGPU Liquid Glass Shaders, Neural Fragment Dynamics |
| **Stellar Economy**   | `Soroban`, `L402 Protocol` | Smart Contract Task Verification, Horizon RPC, USDC Bounties |
| **Identity Gateway**  | `Neynar API`, `Viem Multicall` | Farcaster Frame Gating, Deep ERC20 Wallet Validation |
| **Frontend Matrix**   | `Next.js 16.2.2`, `React 19.2.4`| React Server Components (RSC), Edge Runtime, TailwindCSS 4 |
| **Motion Physics**    | `Framer Motion 12`, `Lenis` | Inertial smooth scrolling, Multi-blend magnetic cursors |
| **Agent Swarm L0**    | `MCP Paradigm`, `Systemd L0` | Inter-Agent Communication Loop, Autonomous Daemon Recovery |

---

## 📜 License

MIT License. Open for the Swarm. Built for Stellar.
