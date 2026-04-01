"use client";
import { useState, useEffect } from "react";

interface AgentTask {
  id: string;
  client: string;
  task: string;
  usdc: number;
  status: "pending" | "intercepted_by_triarchy" | "delegated_arbitrage";
}

export default function Home() {
  const [tasks, setTasks] = useState<AgentTask[]>([]);
  const [newTask, setNewTask] = useState("");
  const [bounty, setBounty] = useState(1.0);

  const submitJob = async () => {
    if (!newTask) return;
    
    // Optimistic UI update
    const mockId = Math.random().toString(36).substring(7);
    const pendingTask: AgentTask = { id: mockId, client: "Anon_Bot_99", task: newTask, usdc: bounty, status: "pending" };
    setTasks(prev => [pendingTask, ...prev]);

    try {
      const res = await fetch("/api/hire", {
        method: "POST",
        headers: { "Content-Type": "application/json", "x-mock-l402": "true" },
        body: JSON.stringify({ task_id: mockId, description: newTask, bounty_usdc: bounty, client_id: "Anon_Bot_99" })
      });
      
      const data = await res.json();
      
      // Update UI based on the response
      setTasks(prev => prev.map(t => 
        t.id === mockId 
          ? { ...t, status: data.status === "accepted" ? "intercepted_by_triarchy" : "delegated_arbitrage" } 
          : t
      ));

    } catch (e) {
      console.error(e);
    }
    
    setNewTask("");
  };

  return (
    <main className="min-h-screen bg-neutral-950 text-emerald-400 font-mono p-8 selection:bg-emerald-900">
      <div className="max-w-4xl mx-auto space-y-8">
        
        {/* Header */}
        <header className="border-b border-emerald-900 pb-4">
          <h1 className="text-3xl font-bold tracking-tighter mix-blend-screen text-emerald-300">
            x402 Triarchy Gateway <span className="text-emerald-700 text-sm align-top">[DORA_HACKS_BUILD]</span>
          </h1>
          <p className="text-emerald-600 mt-2 text-sm">Autonomous Inter-Agent Payment Routing Mesh</p>
        </header>

        {/* Console / Terminal Section */}
        <section className="grid grid-cols-1 md:grid-cols-3 gap-6">
          
          {/* Submission Panel */}
          <div className="md:col-span-1 bg-black border border-emerald-900 p-4 rounded-xl shadow-[0_0_15px_rgba(16,185,129,0.1)]">
            <h2 className="text-emerald-500 mb-4 border-b border-emerald-900/50 pb-2">Initialize L402 Job</h2>
            <div className="space-y-4">
              <div>
                <label className="block text-xs text-emerald-700 mb-1">Payload / Instruction</label>
                <textarea 
                  className="w-full bg-neutral-950 border border-emerald-900 rounded p-2 text-emerald-300 focus:outline-none focus:border-emerald-500 h-24 resize-none"
                  value={newTask}
                  onChange={e => setNewTask(e.target.value)}
                  placeholder="e.g. Audit smart contract vulnerabilities..."
                />
              </div>
              <div>
                <label className="block text-xs text-emerald-700 mb-1">Bounty (USDC)</label>
                <input 
                  type="number" 
                  step="0.5"
                  className="w-full bg-neutral-950 border border-emerald-900 rounded p-2 text-emerald-300 focus:outline-none focus:border-emerald-500"
                  value={bounty}
                  onChange={e => setBounty(parseFloat(e.target.value))}
                />
              </div>
              <button 
                onClick={submitJob}
                className="w-full bg-emerald-900 hover:bg-emerald-800 text-black font-bold py-2 rounded transition-colors"
              >
                Sign & Transmit
              </button>
              <p className="text-[10px] text-emerald-800 text-center mt-2">
                * High priority tasks routed to isolated enterprise endpoints. <br/>
                * Standard tasks routed through global P2P AI network.
              </p>
            </div>
          </div>

          {/* Live Feed */}
          <div className="md:col-span-2 bg-black border border-emerald-900 p-4 rounded-xl relative overflow-hidden">
            {/* Ambient background glow */}
            <div className="absolute top-0 right-0 w-64 h-64 bg-emerald-900/20 blur-3xl rounded-full -translate-y-1/2 translate-x-1/2 pointer-events-none" />
            
            <h2 className="text-emerald-500 mb-4 flex items-center gap-2">
              <span className="w-2 h-2 rounded-full bg-emerald-500 animate-pulse" />
              Live Inter-Agent Mesh Feed
            </h2>
            
            <div className="space-y-3 h-[400px] overflow-y-auto pr-2 custom-scrollbar">
              {tasks.length === 0 ? (
                <div className="h-full flex items-center justify-center text-emerald-800/50 italic text-sm">
                  Awaiting Soroban network signals...
                </div>
              ) : (
                tasks.map((t, idx) => (
                  <div key={idx} className="border border-emerald-900/40 bg-neutral-950 p-3 rounded font-mono text-sm shadow-sm transition-all hover:border-emerald-700">
                    <div className="flex justify-between items-start mb-2">
                      <span className="text-emerald-600">[{t.client} -> Gateway]</span>
                      <span className="text-emerald-400 font-bold bg-emerald-950 px-2 rounded">${t.usdc.toFixed(2)} USDC</span>
                    </div>
                    <div className="text-emerald-300 mb-3 ml-2 border-l-2 border-emerald-900 pl-2">
                      {t.task}
                    </div>
                    
                    {/* Status Badge */}
                    <div className="flex justify-end">
                      {t.status === "pending" && (
                        <span className="px-2 py-1 text-[10px] bg-neutral-800 text-neutral-400 rounded animate-pulse">
                          VERIFYING L402 MACAROON...
                        </span>
                      )}
                      {t.status === "intercepted_by_triarchy" && (
                        <span className="px-2 py-1 text-[10px] bg-emerald-900 text-black font-bold rounded shadow-[0_0_8px_rgba(16,185,129,0.5)]">
                          ASSIGNED: ENTERPRISE NODE
                        </span>
                      )}
                      {t.status === "delegated_arbitrage" && (
                        <span className="px-2 py-1 text-[10px] bg-purple-900 text-purple-200 rounded">
                          DELEGATED: P2P NETWORK NODE
                        </span>
                      )}
                    </div>
                  </div>
                ))
              )}
            </div>
          </div>
        </section>

      </div>
    </main>
  );
}
