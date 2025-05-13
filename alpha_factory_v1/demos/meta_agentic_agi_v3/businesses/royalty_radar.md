```mermaid
%% RoyaltyRadar 👁️✨ – Meta-Agentic Flow
flowchart TD
  subgraph MetaAgent["RoyaltyRadar.a.agi.eth  🧠  (meta-agent)"]
    ORCH["Coordinator Ω"]
    ORCH -->|spawn| DM[\"DataMinerAgent  📊\\n(dsp adapters)\"]:::agent
    ORCH -->|spawn| CL[\"ClaimAgent  📑\\n(Bayes + LLM)\"]:::agent
    ORCH -->|score, mutate, replace| DM
    ORCH -->|score, mutate, replace| CL
  end

  DM -->|public counts| STORE[\"Lineage & Audit  📜\"]:::store
  CL -->|gap, letter, tx-hash| STORE
  CL -->|€ payout (on-chain)| WALLET[\"Artist Wallet  💎\"]:::val

classDef agent fill:#0f9d58,color:#fff,stroke-width:0;
classDef store fill:#2b2b40,color:#fff,stroke-width:0;
classDef val   fill:#1e88e5,color:#fff,stroke-width:0;
```
