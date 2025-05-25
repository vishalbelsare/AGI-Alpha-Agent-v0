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

```mermaid
%% α‑AGI Insight — Beyond Human Foresight
%% System‑level & Repository Structure overview
%% Place this block inside README.md for live rendering.

%% ------------------------------------------------------------------
%% 1. High‑Level Architecture
%% ------------------------------------------------------------------
graph TD
    %% Core orchestration
    Orchestrator["🧠 Macro‑Sentinel<br/>Orchestrator"]:::core
    
    %% Secure message backbone
    Bus["🔗 Secure Pub/Sub<br/>A2A Bus"]:::bus
    
    Orchestrator <-->|register/heartbeat| Bus
    
    %% Primary agent cluster
    subgraph "Agent Swarm"
        direction LR
        Planning["🗺️ Planning<br/>Agent"]:::agent
        Research["🔎 Research<br/>Agent"]:::agent
        Strategy["🎯 Strategy<br/>Agent"]:::agent
        Market["📈 MarketAnalysis<br/>Agent"]:::agent
        CodeGen["💻 CodeGen<br/>Agent"]:::agent
        Safety["🛡️ Safety Guardian"]:::safety
        Memory["📚 Memory Store"]:::memory
    end
    
    Bus --"A2A envelopes"--> Planning
    Bus --> Research
    Bus --> Strategy
    Bus --> Market
    Bus --> CodeGen
    Bus --> Safety
    Bus --> Memory
    
    %% Simulation engine (invoked by agents)
    SimEngine["⚙️ MATS + Thermo‑Forecast Engine"]:::engine
    CodeGen -->|invoke| SimEngine
    Planning --> SimEngine
    Research --> SimEngine
    Strategy --> SimEngine
    
    %% Interfaces
    subgraph "User Interfaces"
        CLI["💻 Hybrid CLI"]:::ui
        WebUI["🌐 Web Dashboard<br/>(Streamlit / FastAPI + React)"]:::ui
    end
    
    CLI -->|gRPC / local call| Orchestrator
    WebUI -->|REST / WebSocket| Orchestrator
    
    %% External connectors
    Plugins["🔌 Plugin Gateway<br/>(MCP Tools)"]:::plugin
    SimEngine <-->|tool calls| Plugins
    
    %% Data & Audit
    Ledger["🗄️ Append‑only Audit Ledger<br/>(SQLite + Merkle Roots)"]:::data
    Memory --> Ledger
    Safety --> Ledger
    Orchestrator --> Ledger
    
    %% Style definitions
    classDef core fill:#ffd9b3,stroke:#333,stroke-width:1.5px;
    classDef agent fill:#d0e6ff,stroke:#1b4f9c,stroke-width:1.5px;
    classDef safety fill:#ffcccc,stroke:#b22222,stroke-width:1.5px;
    classDef memory fill:#e6ffe6,stroke:#2e8b57,stroke-width:1.5px;
    classDef engine fill:#f0f0f0,stroke:#555,stroke-width:1.5px;
    classDef bus fill:#e0d7ff,stroke:#5d3fd3,stroke-width:1.5px,stroke-dasharray: 5 5;
    classDef ui fill:#fff2b2,stroke:#c38f00,stroke-width:1.5px;
    classDef plugin fill:#f7e6ff,stroke:#663399,stroke-width:1.5px;
    classDef data fill:#cccccc,stroke:#333,stroke-width:1.5px;
    
%% ------------------------------------------------------------------
%% 2. Repository Structure (simplified)
%% ------------------------------------------------------------------
    
    classDiagram
        class alpha_agi_insight_v1 {
            +README.md
            +requirements.txt
            +infrastructure/
            +docs/
            +tests/
            +src/
        }
        alpha_agi_insight_v1 --> src
        src --> orchestrator.py
        src --> utils
        src --> simulation
        src --> interface
        src --> agents
        
        class agents {
            +base_agent.py
            +planning_agent.py
            +research_agent.py
            +strategy_agent.py
            +market_agent.py
            +codegen_agent.py
            +safety_agent.py
            +memory_agent.py
        }
        class simulation {
            +mats.py
            +forecast.py
            +sector.py
        }
        class interface {
            +cli.py
            +web_app.py
            +api_server.py
            +web_client/
        }
        class utils {
            +messaging.py
            +config.py
            +logging.py
        }
```
