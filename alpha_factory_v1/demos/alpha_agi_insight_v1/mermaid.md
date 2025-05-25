```mermaid
%% 🎖️ α‑AGI Insight 👁️✨ — Beyond Human Foresight — Official Demo
%% Comprehensive architecture & workflow diagram (ZERO‑DATA)

flowchart TD
  %% === Core Orchestration Layer ===
  subgraph Core["Core Services"]
    Orchestrator["Macro‑Sentinel<br/>Orchestrator"]
    MessageBus["Secure A2A<br/>pub/sub bus"]
  end

  Orchestrator -- "registry / heartbeat" --> MessageBus
  MessageBus -- "routing / TLS" --> Orchestrator

  %% === Agents Swarm ===
  subgraph AgentsPool["Specialised α‑AGI Agents"]
    PlanningAgent["Planning Agent<br/>(OpenAI SDK)"]
    ResearchAgent["Research Agent<br/>(Google ADK)"]
    StrategyAgent["Strategy Agent"]
    MarketAgent["Market‑Analysis Agent"]
    CodeGenAgent["Code‑Gen Agent"]
    SafetyAgent["Safety‑Guardian Agent"]
    MemoryAgent["Memory / Knowledge<br/>Store Agent"]
  end

  MessageBus <--> PlanningAgent
  MessageBus <--> ResearchAgent
  MessageBus <--> StrategyAgent
  MessageBus <--> MarketAgent
  MessageBus <--> CodeGenAgent
  MessageBus <--> MemoryAgent
  SafetyAgent -- "policy guard" --- MessageBus

  %% === Simulation & Analytics Engines ===
  subgraph Simulation["Zero‑Data Simulation Engines"]
    MATS["Meta‑Agentic Tree Search<br/>(NSGA‑II, Eq. 3)"]
    Forecast["Thermodynamic Forecast<br/>(Eq. 1 trigger)"]
    InnovationPool["Elite Innovation Pool"]
    SectorDB["Sector State DB"]
  end

  PlanningAgent -- "spawn search" --> MATS
  ResearchAgent -- "spawn search" --> MATS
  MATS --> InnovationPool
  InnovationPool --> Forecast
  StrategyAgent --> Forecast
  Forecast --> SectorDB

  %% === User Interfaces ===
  subgraph Interfaces["User‑Facing Interfaces"]
    WebUI["Web UI<br/>(Streamlit / React)"]
    CLI["CLI (Click)"]
  end

  SectorDB --> WebUI
  SectorDB --> CLI
  Users["👤 End Users"] <--> WebUI
  Users <--> CLI

  %% === Storage & Audit ===
  subgraph Storage["Immutable Logs & Artifacts"]
    Ledger["Append‑only Ledger<br/>(SQLite + Merkle→Blockchain)"]
    ContainerRegistry["Container Registry"]
  end

  MessageBus -- "hash‑chain events" --> Ledger
  Orchestrator -- "push images" --> ContainerRegistry

  %% === Deployment & Ops ===
  subgraph DevOps["Packaging & Deployment"]
    DockerCompose["Docker‑Compose"]
    HelmChart["Helm (K8s)"]
    Terraform["Terraform<br/>(GCP / AWS)"]
  end

  ContainerRegistry --> DockerCompose
  ContainerRegistry --> HelmChart
  ContainerRegistry --> Terraform

  %% === Offline / Air‑gapped Mode ===
  subgraph OfflineMode["Resource‑Adaptive Runtime"]
    LocalLLM["Local LLM Runtime<br/>(Llama‑3 / GPT‑Neo)"]
  end
  LocalLLM -. "inference" .-> PlanningAgent
  LocalLLM -. "inference" .-> StrategyAgent
  LocalLLM -. "code eval" .-> CodeGenAgent

  %% Styling
  classDef core fill:#e0f7ff,stroke:#0288d1,stroke-width:1px;
  classDef agents fill:#e8f5e9,stroke:#2e7d32,stroke-width:1px;
  classDef sim fill:#fff3e0,stroke:#f57c00,stroke-width:1px;
  classDef iface fill:#f3e5f5,stroke:#7b1fa2,stroke-width:1px;
  classDef storage fill:#ede7f6,stroke:#512da8,stroke-width:1px;
  classDef devops fill:#eceff1,stroke:#455a64,stroke-width:1px;
  classDef offline fill:#ffebee,stroke:#c62828,stroke-width:1px;

  class Core,Orchestrator,MessageBus core;
  class AgentsPool,PlanningAgent,ResearchAgent,StrategyAgent,MarketAgent,CodeGenAgent,SafetyAgent,MemoryAgent agents;
  class Simulation,MATS,Forecast,InnovationPool,SectorDB sim;
  class Interfaces,WebUI,CLI iface;
  class Storage,Ledger,ContainerRegistry storage;
  class DevOps,DockerCompose,HelmChart,Terraform devops;
  class OfflineMode,LocalLLM offline;
```


# 🎖️ α‑AGI Insight 👁️✨ — Beyond Human Foresight — Official Demo  
### Production‑Grade System – Mermaid Specification
> **Note:** Copy‑paste the following Mermaid blocks into your README.md (or any Mermaid‑enabled renderer) to obtain interactive architecture, repository, and DevOps diagrams.

---

## 1. End‑to‑End System Architecture
```mermaid
%% α‑AGI Insight – High‑Level Architecture
graph TD
    %% ===== User Interfaces =====
    subgraph UI[User Interfaces]
        CLI["CLI (Click)"]
        WebUI["Web UI (Streamlit / React+FastAPI)"]
    end

    %% ===== Orchestrator & Bus =====
    subgraph ORCH[Macro‑Sentinel Orchestrator]
        Orchestrator["Orchestrator<br/>(Agent Registry • Heartbeats • Scheduling)"]
        A2ABus[["Secure A2A Message Bus<br/>(gRPC pub/sub, TLS)"]]
        Ledger["Audit Ledger<br/>(SQLite + Merkle Roots)"]
    end

    %% ===== Core Engine =====
    subgraph CORE[Zero‑Data Simulation Engine]
        MATS["MATS Engine<br/>(NSGA‑II Evolutionary Search)"]
        Forecast["Disruption Forecaster<br/>(Thermodynamic Trigger)"]
        Memory["Shared Memory Store"]
    end

    %% ===== Agents =====
    subgraph AGENTS[α‑AGI Agents]
        Planning["Planning Agent"]
        Research["Research Agent"]
        Strategy["Strategy Agent"]
        Market["Market Analysis Agent"]
        CodeGen["CodeGen Agent"]
        Safety["Safety Guardian Agent"]
    end

    %% ===== Tool Layer =====
    subgraph TOOLS[Tooling & Plugins]
        MCP["Model Context Protocol Adapter"]
        Plugins["Safe Plugins<br/>(Data • Viz • Persistence)"]
    end

    %% ===== Data / Chain =====
    Blockchain["Public Blockchain<br/>(Solana Testnet)"]

    %% --- Data Flow ---
    CLI-->|REST / gRPC|Orchestrator
    WebUI-->|REST / WebSocket|Orchestrator

    Orchestrator-->|pub/sub|A2ABus
    A2ABus-->|broadcast|Planning
    A2ABus-->|broadcast|Research
    A2ABus-->|broadcast|Strategy
    A2ABus-->|broadcast|Market
    A2ABus-->|broadcast|CodeGen
    A2ABus-->|monitor|Safety

    Safety-->|policy actions|Orchestrator

    %% Agents tool calls
    Planning-->|call|MCP
    Research-->|call|MCP
    Strategy-->|call|MCP
    CodeGen-->|call|MCP
    MCP-->|executes|Plugins

    %% Core Engine links
    Orchestrator-->|invoke|MATS
    MATS-->|elite pool|Forecast
    Forecast-->|writes|Memory
    Agents-->|query/update|Memory

    Orchestrator-->|log hashes|Ledger
    Ledger-->|checkpoint|Blockchain

    %% Result channels
    Forecast-->|stream results|WebUI
    Forecast-->|print summary|CLI
```

---

## 2. Repository Layout
```mermaid
%% Logical Repository Tree (folders collapsed for brevity)
graph LR
    R[alpha_agi_insight_v0/]---README[README.md]
    R---REQ[requirements.txt]

    subgraph SRC[/src]
        SRC_ORCH[orchestrator.py]
        SRC_AGENTS[/agents]
        SRC_SIM[/simulation]
        SRC_INT[/interface]
        SRC_UTIL[/utils]
    end
    R---SRC

    subgraph AGENTS_TREE
        SRC_AGENTS_BASE[base_agent.py]
        SRC_AGENTS_PLAN[planning_agent.py]
        SRC_AGENTS_RES[research_agent.py]
        SRC_AGENTS_STR[strategy_agent.py]
        SRC_AGENTS_MAR[market_agent.py]
        SRC_AGENTS_CODE[codegen_agent.py]
        SRC_AGENTS_SAFE[safety_agent.py]
        SRC_AGENTS_MEM[memory_agent.py]
    end
    SRC_AGENTS---AGENTS_TREE

    subgraph SIM_TREE
        SRC_SIM_MATS[mats.py]
        SRC_SIM_FORE[forecast.py]
        SRC_SIM_SECT[sector.py]
    end
    SRC_SIM---SIM_TREE

    subgraph INT_TREE
        SRC_INT_CLI[cli.py]
        SRC_INT_WEB[web_app.py]
        SRC_INT_API[api_server.py]
        SRC_INT_REACT[/web_client]
    end
    SRC_INT---INT_TREE

    subgraph UTIL_TREE
        SRC_UTIL_MSG[messaging.py]
        SRC_UTIL_CFG[config.py]
        SRC_UTIL_LOG[logging.py]
    end
    SRC_UTIL---UTIL_TREE

    R---TESTS[/tests]
    R---INFRA[/infrastructure]
    R---DOCS[/docs]
```

---

## 3. CI/CD & Deployment Pipeline
```mermaid
flowchart LR
    Dev[Developer Push]-->CI[GitHub Actions / CI]
    CI-->|Unit & Integration Tests|TestPass{All Tests Pass?}
    TestPass-->|Yes|Build[Docker Multi‑Arch Build]
    TestPass-->|No|Fail[Fail Pipeline]

    Build-->Scan[Security Scan (Snyk/Trivy)]
    Scan-->|Pass|PushReg[Push Image to Registry]

    PushReg-->|Tag Release|HelmChart[Helm Package Update]
    HelmChart-->CD[ArgoCD / Flux]

    CD-->|Deploy|K8s[Kubernetes Cluster<br/>(Prod / Staging)]
    K8s-->|Health Checks|Monitor[Prometheus / Grafana]
    Monitor-->|Alerts|Ops[Ops Team]

    K8s-->|Rolling Update Success|Users[End Users]
```

---

## 4. Legend
- **Solid arrows**: primary data/control flow  
- **Dashed arrows**: monitoring / logging / audit paths  
- **Rounded rectangles**: active services or agents  
- **Parallelograms**: data stores or ledgers  
- **Cylinders**: external persistent storage / blockchain  
