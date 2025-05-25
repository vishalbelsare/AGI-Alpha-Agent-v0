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
## Production‑grade System & Repository Blueprint (Mermaid)

Below are **two complementary Mermaid diagrams** that can be embedded directly in
`alpha_factory_v1/demos/alpha_agi_insight_v1/README.md`.

1. **High‑level runtime architecture** – shows how users, interfaces, the
   Orchestrator, specialised agents, the secure A2A bus, the zero‑data MATS
   simulation engine and the audit‑ledger fit together in production.
2. **Repository layout** – a living map of the code‑base so contributors and
   auditors can instantly understand where every responsibility lives.

---

```mermaid
%% =====================================================================
%%  1️⃣  RUNTIME ARCHITECTURE – α‑AGI Insight (Zero‑Data Edition)
%% =====================================================================
flowchart TD
    subgraph C[🟢 Client Tier]
        User([User<br/>(decision‑maker)])
        CLI[[`alpha-insight` CLI]]
        WebUI[[Web&nbsp;UI<br/>(Streamlit / React)]]
        User -- "commands / scripts" --> CLI
        User -- "interactive flows" --> WebUI
    end

    subgraph O[🔵 Orchestration Core]
        Orchestrator[[Macro‑Sentinel<br/>(Orchestrator)]]
        A2A[[Secure A2A&nbsp;Bus<br/>(gRPC + TLS)]]
    end

    subgraph AG[🟣 Specialised α‑AGI Agents]
        Planning[PlanningAgent]
        Research[ResearchAgent]
        Strategy[StrategyAgent]
        Market[MarketAnalysisAgent]
        CodeGen[CodeGenAgent]
        Safety[SafetyGuardian]
        Memory[MemoryAgent]
    end

    subgraph SIM[🟠 Zero‑Data Simulation]
        Engine[[MATS + Thermo‑Forecast Engine]]
    end

    subgraph ST[🟡 Persistence & Audit]
        Ledger[(Append‑only Audit Ledger<br/>+ Merkle roots)]
        KStore[(Shared Knowledge Store)]
    end

    subgraph DEPLOY[⚙️  Deployment Fabric]
        Docker[(Docker / K8s&nbsp;Pods)]
        Helm[(Helm Charts)]
        TF[(Terraform IaC)]
    end

    %% --- Interface paths ---
    CLI --> Orchestrator
    WebUI --> Orchestrator

    %% --- Orchestrator <-> agents ---
    Orchestrator -- "register / heartbeat" --> A2A
    A2A <---> Planning & Research & Strategy & Market & CodeGen & Safety & Memory

    %% --- Simulation calls ---
    Orchestrator -- "spawn run / collect results" --> Engine
    Planning & Research & Strategy & Market & CodeGen --> Engine
    Engine -- "trajectories / KPIs" --> Orchestrator
    Engine -- "sector curves" --> WebUI

    %% --- Persistence ---
    Orchestrator --> Ledger
    Planning & Research & Strategy & Market & CodeGen --> Ledger
    Safety --> Ledger
    Memory --- KStore
    Engine --> KStore

    %% --- Deployment mapping ---
    Docker --> Orchestrator & A2A & Engine & Planning & Research & Strategy & Market & CodeGen & Safety & Memory & WebUI
    Helm --> Docker
    TF --> Docker
```

---

```mermaid
%% =====================================================================
%% 2️⃣  MONOREPO STRUCTURE – alpha_agi_insight_v1
%% =====================================================================
flowchart TD
    R[alpha_agi_insight_v1]:::root
    R --> READM[README.md]
    R --> REQ[requirements.txt]

    subgraph SRCDIR["src/"]:::dir
        SRCDIR --> ORCH[orchestrator.py]
        SRCDIR --> AGENTS[agents/]:::dir
        SRCDIR --> SIMS[simulation/]:::dir
        SRCDIR --> INTF[interface/]:::dir
        SRCDIR --> UTIL[utils/]:::dir
    end
    R --> SRCDIR

    %% agents subtree
    AGENTS --> BASE[base_agent.py]
    AGENTS --> PLA[planning_agent.py]
    AGENTS --> RES[research_agent.py]
    AGENTS --> STR[strategy_agent.py]
    AGENTS --> MAR[market_agent.py]
    AGENTS --> CG[codegen_agent.py]
    AGENTS --> SAF[safety_agent.py]
    AGENTS --> MEM[memory_agent.py]

    %% simulation subtree
    SIMS --> MATS[mats.py]
    SIMS --> FORE[forecast.py]
    SIMS --> SECT[sector.py]

    %% interface subtree
    INTF --> CLI[cli.py]
    INTF --> WAPP[web_app.py]
    INTF --> API[api_server.py]
    INTF --> WEBCL[web_client/]:::dir

    %% utils subtree
    UTIL --> MSG[messaging.py]
    UTIL --> CFG[config.py]
    UTIL --> LOG[logging.py]

    %% top‑level peers
    R --> TESTS[tests/]:::dir
    R --> INFRA[infrastructure/]:::dir
    R --> DOCS[docs/]:::dir

    classDef dir fill:#e6f7ff,stroke:#0284c7,stroke-width:1px;
    classDef root fill:#fffbe6,stroke:#d97706,stroke-width:2px;
```

