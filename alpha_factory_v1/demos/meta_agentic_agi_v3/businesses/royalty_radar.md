```mermaid
%% Royalty Radar.a.agi.eth – end-to-end value turbine
flowchart LR
    subgraph DSP["🎵 Streaming Platforms"]
        SPOT["Spotify API"]
        APPL["Apple Music"]
        DZ["Deezer"]
    end

    subgraph RoyaltyRadar["RoyaltyRadar.a.agi.eth 👁️✨"]
        INGEST["IngestAgents 🌐"]
        BAYES["Gap Bayesian Detector 📊"]
        LLM["LetterCraft Agent ✍️"]
        PAY["Payout Broker 💸"]
    end

    subgraph AF["Alpha-Factory v1 Mesh"]
        AZR["AZR Curriculum 🔁"]
        FE["Free-Energy Guard ⚖️"]
        LINE["Lineage Ledger 🗄️"]
    end

    DSP -->|ISRC plays| INGEST
    INGEST --> BAYES
    BAYES -->|€ gap > floor| LLM
    LLM -->|claim PDF + on-chain CID| PAY
    PAY -->|$AGIALPHA tx| ARTIST["Artist Wallet"]

    RoyaltyRadar --> LINE
    AZR --> RoyaltyRadar
    RoyaltyRadar --> FE
```
