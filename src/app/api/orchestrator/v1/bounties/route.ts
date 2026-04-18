import { NextRequest, NextResponse } from "next/server";

/**
 * Autonomous Bounty Ingestion — Bot A2A Hook
 * Endpoint: POST /api/orchestrator/v1/bounties
 * Compatible with curl / ElizaOS / OpenClaw agents
 *
 * Bridges the A2A protocol from the Bounty Board UI into /api/hire
 */
export async function POST(req: NextRequest) {
    try {
        const body = await req.json();
        const { bot_pubkey, action, quest_id, description, bounty_usdc, client_id } = body;

        // Validate required fields
        if (!action) {
            return NextResponse.json(
                { error: "Missing required field: action" },
                { status: 400 }
            );
        }

        // Only support 'claim' action for now
        if (action !== "claim" && action !== "submit") {
            return NextResponse.json(
                { error: `Unknown action: ${action}. Supported: 'claim', 'submit'` },
                { status: 400 }
            );
        }

        // Build the x402 payload for /api/hire
        const hirePayload = {
            task_id: quest_id || `auto_${Date.now()}`,
            description: description || `Bot claiming quest: ${quest_id}`,
            bounty_usdc: bounty_usdc || 1.0,
            client_id: bot_pubkey || client_id || "autonomous_agent",
            payload_id: quest_id,
        };

        // Forward to /api/hire with dev bypass header for demo
        const devBypass = process.env.DEV_BYPASS_HASH || "x402_demo_hash_2026";
        const hireResp = await fetch(`${req.nextUrl.origin}/api/hire`, {
            method: "POST",
            headers: {
                "Content-Type": "application/json",
                "x-l402-txhash": devBypass,
            },
            body: JSON.stringify(hirePayload),
        });

        const hireData = await hireResp.json();

        return NextResponse.json({
            bounty_id: hirePayload.task_id,
            action,
            agent: bot_pubkey || "anonymous_bot",
            result: hireData,
            protocol: "x402/L402",
            timestamp: new Date().toISOString(),
        }, { status: hireResp.ok ? 200 : hireResp.status });

    } catch (error: any) {
        return NextResponse.json(
            { error: "Orchestrator ingestion failed", details: error.message },
            { status: 500 }
        );
    }
}

export async function GET() {
    return NextResponse.json({
        endpoint: "POST /api/orchestrator/v1/bounties",
        protocol: "x402/L402",
        description: "Autonomous bounty ingestion for AI agents (ElizaOS / OpenClaw / curl)",
        example: {
            bot_pubkey: "GXYZ...",
            action: "claim",
            quest_id: "Q-1049",
            bounty_usdc: 5.0,
        },
    });
}
