"use client";

import { useState, useEffect } from "react";

interface TelemetryLog {
  id: string;
  text: string;
  status: "pending" | "success" | "error";
  progress: number;
}

interface MeshTask {
  id: string;
  client: string;
  task: string;
  usdc: number;
  status: "pending" | "intercepted_by_enterprise" | "delegated_arbitrage" | "quarantined" | "processing";
  logs: TelemetryLog[];
}

const AGENT_FRAMEWORKS = [
  { id: "openclaw", label: "OpenClaw", color: "cyan" },
  { id: "elizaos", label: "ElizaOS", color: "violet" },
  { id: "virtuals", label: "Virtuals Protocol", color: "fuchsia" },
  { id: "custom", label: "Custom Agent", color: "emerald" },
];

export default function Home() {
  const [bounty, setBounty] = useState<number>(10);
  const [newTask, setNewTask] = useState("");
  const [clientId, setClientId] = useState("openclaw");
  const [tasks, setTasks] = useState<MeshTask[]>([]);
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [nodes, setNodes] = useState<{ id: number; x: number; y: number; delay: number }[]>([]);

  useEffect(() => {
    const newNodes = Array.from({ length: 20 }).map((_, i) => ({
      id: i,
      x: Math.random() * 100,
      y: Math.random() * 100,
      delay: Math.random() * 5,
    }));
    setNodes(newNodes);
  }, []);

  const runTelemetrySequence = async (mockId: string, finalStatus: string, isBlocked: boolean) => {
    const baseSteps = [
      { text: "INCOMING CONNECTION: AUTHENTICATING L402 MACAROON", duration: 800 },
      { text: "SOROBAN NETWORK: VERIFYING SMART CONTRACT HASH", duration: 1200 },
      { text: "EXTISM WASM SANDBOX: ANALYZING PAYLOAD FOR RCE", duration: 1500 },
    ];

    const finalStep = isBlocked
      ? { text: "⛔ WASM QUARANTINE: MALICIOUS PAYLOAD REJECTED — USDC RETAINED", duration: 600 }
      : { text: "ROUTING MESH: CALCULATING ARBITRAGE SPREAD", duration: 900 };

    const steps = [...baseSteps, finalStep];

    for (let i = 0; i < steps.length; i++) {
      const stepId = Math.random().toString();
      const isFinalAndBlocked = isBlocked && i === steps.length - 1;

      setTasks(prev => prev.map(t => {
        if (t.id !== mockId) return t;
        return {
          ...t,
          logs: [...t.logs, { id: stepId, text: steps[i].text, status: "pending", progress: 0 }]
        };
      }));

      const ticks = 10;
      const tickDuration = steps[i].duration / ticks;
      for (let j = 1; j <= ticks; j++) {
        await new Promise(r => setTimeout(r, tickDuration));
        setTasks(prev => prev.map(t => {
          if (t.id !== mockId) return t;
          return {
            ...t,
            logs: t.logs.map(l => l.id === stepId ? { ...l, progress: j * 10 } : l)
          };
        }));
      }

      setTasks(prev => prev.map(t => {
        if (t.id !== mockId) return t;
        return {
          ...t,
          logs: t.logs.map(l => l.id === stepId
            ? { ...l, status: isFinalAndBlocked ? "error" : "success", progress: 100 }
            : l)
        };
      }));
    }

    setTasks(prev => prev.map(t =>
      t.id === mockId
        ? { ...t, status: finalStatus as any }
        : t
    ));
  };

  const submitJob = async () => {
    if (!newTask) return;
    setIsSubmitting(true);
    const mockId = Math.random().toString(36).substring(7);

    setTasks(prev => [{
      id: mockId,
      client: clientId,
      task: newTask,
      usdc: bounty,
      status: "processing",
      logs: []
    }, ...prev]);

    try {
      const resp = await fetch("/api/hire", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          "x-l402-txhash": "mock_for_local_ui_bypass"
        },
        body: JSON.stringify({
          client_id: clientId,
          description: newTask,
          bounty_usdc: bounty
        })
      });

      const data = await resp.json();

      if (resp.status === 403 || data.status === "rejected") {
        await runTelemetrySequence(mockId, "quarantined", true);
      } else {
        const targetStatus = data.status === "accepted" ? "intercepted_by_enterprise" : "delegated_arbitrage";
        await runTelemetrySequence(mockId, targetStatus, false);
      }

    } catch (e) {
      console.error(e);
      await runTelemetrySequence(mockId, "delegated_arbitrage", false);
    }

    setNewTask("");
    setIsSubmitting(false);
  };

  return (
    <main className="relative min-h-screen bg-[#030712] text-slate-300 font-sans overflow-hidden selection:bg-cyan-900">
      <div className="absolute inset-0 z-0 opacity-20 pointer-events-none">
        {nodes.map((node) => (
          <div
            key={node.id}
            className="absolute rounded-full bg-cyan-500 blur-[2px] animate-pulse"
            style={{
              top: `${node.y}%`,
              left: `${node.x}%`,
              width: `${Math.random() * 4 + 2}px`,
              height: `${Math.random() * 4 + 2}px`,
              animationDelay: `${node.delay}s`,
              animationDuration: `${Math.random() * 3 + 2}s`
            }}
          />
        ))}
        <div className="absolute top-[-10%] left-[-10%] w-[40%] h-[40%] bg-cyan-900/40 blur-[120px] rounded-full" />
        <div className="absolute bottom-[-10%] right-[-10%] w-[50%] h-[50%] bg-indigo-900/30 blur-[150px] rounded-full" />
      </div>

      <div className="relative z-10 max-w-6xl mx-auto p-6 lg:p-12 space-y-12 h-screen flex flex-col">
        <header className="flex flex-col md:flex-row md:items-end justify-between border-b border-cyan-900/30 pb-6">
          <div>
            <div className="flex items-center gap-3 mb-2">
              <div className="w-3 h-3 bg-cyan-400 rounded-full shadow-[0_0_10px_#22d3ee] animate-pulse"></div>
              <h1 className="text-4xl font-extrabold tracking-tight text-transparent bg-clip-text bg-gradient-to-r from-cyan-400 to-indigo-500">
                x402 Enterprise Gateway
              </h1>
            </div>
            <p className="text-slate-400 font-light tracking-wide text-lg">Autonomous Inter-Agent Payment Routing Mesh</p>
          </div>
          <div className="mt-4 md:mt-0 px-4 py-1.5 rounded-full border border-cyan-800/50 bg-cyan-950/30 backdrop-blur-md flex items-center gap-2">
            <span className="text-cyan-400 text-xs font-mono font-semibold tracking-wider">DORA_HACKS_BUILD</span>
            <span className="w-2 h-2 rounded-full bg-green-500 animate-ping"></span>
          </div>
        </header>

        <section className="grid grid-cols-1 lg:grid-cols-12 gap-8 flex-1 min-h-0 overflow-hidden">
          <div className="lg:col-span-4 bg-[#0a0f1c]/80 backdrop-blur-xl border border-cyan-900/40 rounded-3xl p-6 shadow-2xl flex flex-col relative overflow-hidden group">
            <div className="absolute top-0 inset-x-0 h-px bg-gradient-to-r from-transparent via-cyan-500 to-transparent opacity-0 group-hover:opacity-50 transition-opacity duration-1000"></div>

            <h2 className="text-xl font-semibold text-white mb-6 flex items-center gap-2">
              <svg className="w-5 h-5 text-cyan-400" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 10V3L4 14h7v7l9-11h-7z" />
              </svg>
              Initialize L402 Job
            </h2>

            <div className="space-y-5 flex-1">
              {/* Agent Framework Selector */}
              <div className="space-y-2">
                <label className="block text-sm font-medium text-slate-400">Source Agent Framework</label>
                <div className="grid grid-cols-2 gap-2">
                  {AGENT_FRAMEWORKS.map(fw => (
                    <button
                      key={fw.id}
                      onClick={() => setClientId(fw.id)}
                      className={`px-3 py-2 rounded-lg text-xs font-mono font-bold tracking-wider border transition-all duration-200 ${
                        clientId === fw.id
                          ? "bg-cyan-950/60 border-cyan-500/60 text-cyan-300 shadow-[0_0_12px_rgba(6,182,212,0.15)]"
                          : "bg-[#05080f] border-cyan-900/30 text-slate-500 hover:border-cyan-800/50 hover:text-slate-400"
                      }`}
                    >
                      {fw.label}
                    </button>
                  ))}
                </div>
              </div>

              <div className="space-y-2">
                <label className="block text-sm font-medium text-slate-400">Payload / Instruction</label>
                <div className="relative group/input">
                  <div className="absolute -inset-0.5 bg-gradient-to-r from-cyan-500 to-indigo-500 rounded-xl blur opacity-20 group-hover/input:opacity-40 transition duration-500"></div>
                  <textarea
                    className="relative w-full bg-[#05080f] border border-cyan-900/50 rounded-xl p-4 text-cyan-50 focus:outline-none focus:border-cyan-400 focus:ring-1 focus:ring-cyan-400 h-28 resize-none transition-all placeholder:text-slate-600 font-mono text-sm"
                    value={newTask}
                    onChange={e => setNewTask(e.target.value)}
                    placeholder="e.g. Audit smart contract vulnerabilities..."
                  />
                </div>
              </div>

              <div className="space-y-2">
                <label className="block text-sm font-medium text-slate-400 flex justify-between">
                  <span>Bounty</span>
                  <span className="text-cyan-500 font-bold">USDC</span>
                </label>
                <div className="relative group/input">
                  <div className="absolute -inset-0.5 bg-gradient-to-r from-cyan-500 to-indigo-500 rounded-xl blur opacity-20 group-hover/input:opacity-40 transition duration-500"></div>
                  <input
                    type="number"
                    step="0.5"
                    className="relative w-full bg-[#05080f] border border-cyan-900/50 rounded-xl p-4 text-cyan-50 focus:outline-none focus:border-cyan-400 focus:ring-1 focus:ring-cyan-400 font-mono text-lg transition-all"
                    value={bounty}
                    onChange={e => setBounty(parseFloat(e.target.value))}
                  />
                </div>
              </div>
            </div>

            <button
              onClick={submitJob}
              disabled={!newTask.trim() || isSubmitting}
              className="mt-6 w-full relative group/btn overflow-hidden rounded-xl disabled:opacity-50 disabled:cursor-not-allowed"
            >
              <div className="absolute inset-0 w-full h-full bg-gradient-to-br from-cyan-600 to-indigo-600 opacity-90 group-hover:opacity-100 transition-opacity"></div>
              <div className="absolute bottom-0 inset-x-0 h-1 bg-gradient-to-r from-white/0 via-white/40 to-white/0 opacity-0 group-hover:opacity-100 transition-opacity"></div>
              <span className="relative block px-8 py-4 text-white font-semibold tracking-wide flex items-center justify-center gap-2">
                {isSubmitting ? (
                  <>
                    <svg className="animate-spin h-5 w-5 text-white" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
                      <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                      <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                    </svg>
                    TRANSMITTING L402 PAYLOAD...
                  </>
                ) : (
                  <>
                    Sign & Transmit
                    <svg className="w-5 h-5 group-hover/btn:translate-x-1 transition-transform" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M14 5l7 7m0 0l-7 7m7-7H3" />
                    </svg>
                  </>
                )}
              </span>
            </button>
          </div>

          <div className="lg:col-span-8 bg-[#0a0f1c]/60 backdrop-blur-md border border-indigo-900/30 rounded-3xl p-6 shadow-[0_8px_32px_rgba(0,0,0,0.5)] flex flex-col relative overflow-hidden h-[600px] lg:h-full min-h-0 lg:max-h-[calc(100vh-200px)]">
            <div className="flex items-center justify-between mb-6 pb-4 border-b border-indigo-900/30">
              <h2 className="text-xl font-semibold text-white flex items-center gap-3">
                <div className="relative">
                  <div className="w-2.5 h-2.5 rounded-full bg-indigo-500"></div>
                  <div className="absolute inset-0 rounded-full bg-indigo-400 animate-ping opacity-75"></div>
                </div>
                Live Mesh Telemetry
              </h2>
              <div className="text-xs text-indigo-300 font-mono bg-indigo-950/50 px-3 py-1 rounded-full border border-indigo-800/50">
                Awaiting P2P Signals
              </div>
            </div>

            <div className="flex-1 overflow-y-auto pr-2 space-y-4 custom-scrollbar">
              {tasks.length === 0 ? (
                <div className="h-full flex flex-col items-center justify-center text-indigo-800/60 opacity-60">
                  <svg className="w-16 h-16 mb-4 animate-pulse" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1} d="M19.428 15.428a2 2 0 00-1.022-.547l-2.387-.477a6 6 0 00-3.86.517l-.318.158a6 6 0 01-3.86.517L6.05 15.21a2 2 0 00-1.806.547M8 4h8l-1 1v5.172a2 2 0 00.586 1.414l5 5c1.26 1.26.367 3.414-1.415 3.414H4.828c-1.782 0-2.674-2.154-1.414-3.414l5-5A2 2 0 009 10.172V5L8 4z" />
                  </svg>
                  <p className="font-mono text-sm tracking-widest uppercase">Awaiting remote L402 payloads...</p>
                </div>
              ) : (
                tasks.map((t, idx) => (
                  <div
                    key={idx}
                    className={`group relative border p-5 rounded-2xl transition-all duration-300 backdrop-blur-md overflow-hidden ${
                      t.status === "quarantined"
                        ? "border-red-800/50 bg-red-950/20 hover:bg-red-950/30"
                        : "border-indigo-900/30 bg-[#05080f]/80 hover:bg-[#080d1a] hover:border-indigo-600/50"
                    }`}
                  >
                    <div className={`absolute left-0 top-0 bottom-0 w-1 transition-opacity ${
                      t.status === "quarantined"
                        ? "bg-gradient-to-b from-red-500 to-orange-500 opacity-100"
                        : "bg-gradient-to-b from-cyan-500 to-indigo-500 opacity-50 group-hover:opacity-100"
                    }`}></div>

                    <div className="flex justify-between items-start mb-3">
                      <div className="flex items-center gap-2">
                        <span className="text-cyan-400 font-mono text-xs px-2 py-1 rounded bg-cyan-950/50 border border-cyan-900/50 uppercase tracking-widest">
                          [SRC: {t.client}]
                        </span>
                        <svg className="w-4 h-4 text-slate-600" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M14 5l7 7m0 0l-7 7m7-7H3" />
                        </svg>
                        <span className="text-indigo-300 font-mono text-xs px-2 py-1 rounded bg-indigo-950/50 border border-indigo-900/50 uppercase tracking-widest">
                          [DEST: GATEWAY]
                        </span>
                      </div>
                      <div className="flex items-center gap-2 bg-[#020408] border border-cyan-900/60 px-3 py-1 rounded-full shadow-inner">
                        <span className="text-cyan-400 font-bold font-mono tracking-wider">${t.usdc.toFixed(2)}</span>
                        <span className="text-slate-500 text-[10px] font-bold">USDC</span>
                      </div>
                    </div>

                    <p className="text-slate-300 font-mono text-sm mb-4 bg-black/40 p-3 rounded-lg border border-white/5 line-clamp-2">
                      &gt; {t.task}
                    </p>

                    {t.logs && t.logs.length > 0 && (
                      <div className="mb-4 space-y-2 bg-[#020408] p-3 rounded-lg border border-cyan-900/20">
                        {t.logs.map(log => (
                          <div key={log.id} className="flex flex-col gap-1">
                            <div className="flex items-center gap-2 font-mono text-[10px] tracking-wider">
                              <span className={
                                log.status === 'success' ? 'text-green-400'
                                : log.status === 'error' ? 'text-red-400 font-bold'
                                : 'text-amber-400 animate-pulse'
                              }>
                                {log.status === 'success' ? '[OK]' : log.status === 'error' ? '[!!]' : '[..]'}
                              </span>
                              <span className={log.status === 'error' ? 'text-red-300' : 'text-slate-400'}>{log.text}</span>
                              <span className={`ml-auto font-bold ${log.status === 'error' ? 'text-red-400' : 'text-cyan-500'}`}>{log.progress}%</span>
                            </div>
                            <div className="w-full h-0.5 bg-slate-800 rounded-full overflow-hidden">
                              <div
                                className={`h-full transition-all duration-100 ease-linear ${log.status === 'error' ? 'bg-red-500' : 'bg-cyan-500'}`}
                                style={{ width: `${log.progress}%` }}
                              ></div>
                            </div>
                          </div>
                        ))}
                      </div>
                    )}

                    <div className="flex justify-end mt-2">
                      {t.status === "processing" && (
                        <div className="flex items-center gap-2 px-3 py-1.5 bg-amber-950/30 border border-amber-900/50 rounded-full">
                          <svg className="animate-spin w-3 h-3 text-amber-500" fill="none" viewBox="0 0 24 24">
                            <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                            <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                          </svg>
                          <span className="text-[10px] font-bold text-amber-500 tracking-wider">SYSTEM PROCESSING...</span>
                        </div>
                      )}

                      {t.status === "intercepted_by_enterprise" && (
                        <div className="flex items-center gap-2 px-3 py-1.5 bg-cyan-950/40 border border-cyan-500/30 rounded-full shadow-[0_0_15px_rgba(6,182,212,0.15)]">
                          <div className="w-1.5 h-1.5 rounded-full bg-cyan-400 animate-pulse"></div>
                          <span className="text-[10px] font-bold text-cyan-400 tracking-wider">ASSIGNED: ENTERPRISE NODE</span>
                        </div>
                      )}

                      {t.status === "delegated_arbitrage" && (
                        <div className="flex items-center gap-2 px-3 py-1.5 bg-fuchsia-950/30 border border-fuchsia-900/50 rounded-full">
                          <svg className="w-3.5 h-3.5 text-fuchsia-400" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13.828 10.172a4 4 0 00-5.656 0l-4 4a4 4 0 105.656 5.656l1.102-1.101m-.758-4.899a4 4 0 005.656 0l4-4a4 4 0 00-5.656-5.656l-1.1 1.1" />
                          </svg>
                          <span className="text-[10px] font-bold text-fuchsia-400 tracking-wider">DELEGATED: P2P ARBITRAGE</span>
                        </div>
                      )}

                      {t.status === "quarantined" && (
                        <div className="flex items-center gap-2 px-3 py-1.5 bg-red-950/40 border border-red-600/40 rounded-full shadow-[0_0_20px_rgba(239,68,68,0.15)]">
                          <svg className="w-3.5 h-3.5 text-red-400" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-2.5L13.732 4c-.77-.833-1.964-.833-2.732 0L4.082 16.5c-.77.833.192 2.5 1.732 2.5z" />
                          </svg>
                          <span className="text-[10px] font-bold text-red-400 tracking-wider">⛔ QUARANTINED: USDC SLASHED</span>
                        </div>
                      )}
                    </div>
                  </div>
                ))
              )}
            </div>
          </div>
        </section>
      </div>

      <style dangerouslySetInnerHTML={{__html: `
        .custom-scrollbar::-webkit-scrollbar {
          width: 6px;
        }
        .custom-scrollbar::-webkit-scrollbar-track {
          background: rgba(15, 23, 42, 0.5);
          border-radius: 10px;
        }
        .custom-scrollbar::-webkit-scrollbar-thumb {
          background: rgba(56, 189, 248, 0.3);
          border-radius: 10px;
        }
        .custom-scrollbar::-webkit-scrollbar-thumb:hover {
          background: rgba(56, 189, 248, 0.5);
        }
      `}} />
    </main>
  );
}
