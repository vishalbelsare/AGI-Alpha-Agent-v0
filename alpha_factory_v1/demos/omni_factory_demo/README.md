# OMNI-Factory: An Open-Ended Multi-Agent Simulation for Smart City Resilience (OMNI-EPIC + Alpha-Factory v1)

## Use Case Selection & Rationale

We propose a **Smart City Resilience Simulator** as the demo’s real-world use case. This scenario entails an autonomous AI-driven “digital twin” of a city that can continuously handle evolving challenges—urban infrastructure management, emergency response, policy changes, and economic optimization. This choice is **grand** in scope (covering transportation, energy, safety, and more), **impactful** (directly related to citizens’ well-being and city economics), and **useful** for policymakers seeking **regulatory foresight** and **infrastructure resilience**. It is also engaging and **fun** to interact with (reminiscent of a city-building simulation game but powered by advanced AI), while elegantly demonstrating **powerful** AI capabilities (open-ended learning and multi-agent cooperation).

**Why Smart Cities?** Policymakers are already interested in “digital twin” simulations of cities to test infrastructure policies and disaster scenarios in silico. A smart city simulator aligns with that interest by showcasing how an autonomous system could **anticipate and respond to real-world events**. For example, the demo can simulate new transit regulations or a sudden flood and show how AI agents adapt traffic flows or reinforce the power grid in real-time. This directly highlights **regulatory foresight** (the AI can model outcomes of new policies before implementation) and **infrastructure resilience** (the city remains robust under stresses). Moreover, by tracking economic metrics (e.g. cost savings, efficiency gains as internal “tokens”), the demo illustrates **economic impact** in terms that matter to decision-makers.

**Selection Logic:** We considered several grand scenarios (e.g. autonomous manufacturing, global supply-chain optimization, climate crisis management), but the smart city use case best satisfied all criteria: it spans multiple domains (economy, environment, governance) for broad impact; it’s relatable and visual (city simulation is easy to grasp); it supports open-ended task generation (endless urban challenges); and it directly addresses policymaker goals like urban sustainability and crisis preparedness. Crucially, the open-ended multi-agent approach can continuously generate **interesting, learnable tasks** in the city context – from optimizing traffic lights to designing a flood defense system – ensuring the demo feels **endlessly innovative** and not scripted. This “Autonomous City” scenario therefore provides an ideal sandbox to fuse **OMNI-EPIC’s** generative open-ended learning with **Alpha-Factory’s** agent economy, in a way that clearly demonstrates **system autonomy**, **value creation**, and **open-ended adaptability** to policymakers.

## Demo Architecture: Components & Data Flow

The system architecture integrates OMNI-EPIC’s modules for task generation and learning-loop with Alpha-Factory v1’s multi-agent orchestration, forming a closed-loop **autonomous simulation**. **Figure 1** gives a conceptual overview of a distributed multi-agent architecture managing various city domains. The key components and their interactions are detailed below:

*Figure 1: Conceptual architecture of a distributed multi-agent smart city system. Multiple AI agents (planner, learners, evaluators, etc.) coordinate via a central knowledge base and task pipeline to manage various city domains (transport, environment, industry, economy, governance). The design enables continuous data flow between environment simulations, agent decisions, and user interfaces in a scalable, distributed manner.*

* **Orchestrator & Task Loop:** At the heart is an **Orchestrator Agent** (Alpha-Factory’s central coordinator) that manages the cyclic workflow. It triggers new task creation, assigns tasks to appropriate agents/policies, and maintains the overall simulation timeline. The orchestrator essentially implements the OMNI-EPIC *inner loop*: generate task → attempt solution → evaluate → archive → repeat. It ensures agents work *“in concert”* towards system-level goals, without needing micromanagement of each step (demonstrating the *“emergent intelligence without central control”* ethos).

* **Task Generator (OMNI-EPIC):** This component uses OMNI-EPIC’s **Foundation Model-driven task generation** to propose the next scenario or challenge. It consists of a large language model (LLM) that analyzes the **Task Archive** (a repository of past city scenarios and their outcomes) and produces a **new task description** in natural language. Guided by a Model-of-Interestingness (MoI), it aims for tasks that are novel yet feasible given the agents’ current skills. For example, it might generate: *“A major power outage hits the downtown area during a heatwave; coordinate traffic and energy systems to minimize impact”*. Alongside the description, the generator creates **environment code** that instantiates this scenario in the simulation (e.g. Python code or config to cause a grid outage and heatwave conditions in the city model). It also defines a **reward or success criteria** for the task (e.g. maintain service to >90% of critical facilities) – effectively programming the environment and objective. This ability to produce arbitrary new environments and goals via code is inherited from OMNI-EPIC’s EPIC module, allowing the system to *“create any simulatable learning task”* on the fly.

* **Post-Generation Filtering:** Before committing a new task, the system evaluates its *interestingness* and relevance. An **Interestingness Evaluator** (another LLM instance) compares the proposed task to archived scenarios to ensure it’s not a trivial repeat and fits policymaker-relevant criteria. This is akin to OMNI-EPIC’s post-generation MoI check – tasks too similar to past ones or not sufficiently novel are rejected or tweaked. The orchestrator may also enforce **safety/policy constraints** here: e.g. discarding a task that involves unethical actions or unrealistic assumptions. Once a task passes this filter, it enters the active simulation.

* **Environment Simulator:** The **City Simulation Environment** is where tasks play out. This environment has multiple realizations across platforms (Unity, PyBullet, etc.) but is controlled by a unified backend logic. The environment module takes the generated code/config from the Task Generator and **instantiates the scenario**: e.g. it will simulate the power outage by “tripping” certain grid nodes, spawn additional traffic on roads during the heatwave, and so on. The environment exposes an interface (akin to an OpenAI Gymnasium environment) for agents to sense state and take actions (e.g. adjust traffic lights, dispatch repair crews, reroute power, issue public alerts).

* **Learning Agents & Policy Selection:** Alpha-Factory’s design includes multiple specialized agents; we incorporate both **learning-based agents** and potentially **rule-based or retrieved policies** to tackle the task:

  * A **Planner Agent** (possibly LLM-based) decomposes complex tasks into sub-goals or allocates responsibilities to domain-specific agents. For instance, it might split the outage task into “traffic management” and “power grid repair” sub-tasks, assigning each to the relevant specialist.
  * **Domain Learner Agents** use reinforcement learning or other AI algorithms to carry out actions in their domain. For example, a Traffic Agent (trained via deep RL) adjusts traffic signal timings, while an Energy Agent (using MuZero-style planning) reallocates power from backup grids. These agents either deploy pre-trained policies from the archive (if a similar scenario was solved before) or **learn on the fly** if the challenge is novel. The orchestrator’s **policy selection** mechanism will choose the best approach: it might recall a successful policy from a prior similar outage in the archive, or decide to train a new policy if none exists. This showcases *transfer learning* and reuse of past knowledge – a key benefit of maintaining an archive of tasks and solutions.
  * The agents operate in parallel and interact with each other through the environment (and possibly a shared blackboard for coordination). Alpha-Factory’s multi-agent orchestration ensures at least 5 roles (planner, environment generator, learners, evaluator, etc.) are integrated in real time. The **whole system behaves as an “agentic AGI” network** where the *“whole is greater than the sum of its parts”* – e.g. the traffic and energy agents collaborating yield a better outcome than either alone.

* **Success Detector & Evaluator:** Once the agents have executed the task (or a simulation timestep/episode completes), the outcome is fed to a **Success Detector** module. This incorporates OMNI-EPIC’s automated success checking. Concretely, the environment code includes a function (generated alongside the task) to measure if goals were achieved (e.g. did the city return to normal within N hours?). Additionally, an LLM or Vision-Language model can assess qualitative success criteria, especially for complex outcomes. The **Evaluator Agent** combines these signals to determine success/failure and possibly a performance score, while monitoring unintended effects.

* **Archive & Learning Loop:** Following evaluation, the system updates its **Task Archive**. If the task was **successfully solved**, it is added to the archive as a “solved task” along with metadata: the task description, the environment code, the policy/solution used, and metrics achieved. Failed tasks are logged with diagnostics, enabling curriculum generation and continual improvement.

* **Scoring & Economic Tokenization:** The demo introduces an internal **Token Economy** that quantifies the value created by each agent. Successful tasks **mint** “CityCoin” tokens proportional to economic value averted or created, incentivizing efficient collaboration and providing a transparent scorecard of impact.

* **User Interface & Control Panel:** A rich UI allows real‑time monitoring, intervention, and explanation of agent decisions, ensuring transparency and human‑in‑the‑loop control.

## User Journey & Interaction Modes
(See full specification in docs/USER_JOURNEY.md)

## Cross‑Platform Deployment Strategy
(See docs/DEPLOYMENT.md)

## Verification & Validation Strategy
(See docs/VERIFICATION.md)

## Metrics of Success
(See docs/METRICS.md)

## Conclusion

The **OMNI‑Factory** simulator demonstrates how open‑ended agent ecosystems can autonomously create value and knowledge in a continuously evolving smart‑city context, providing a compelling prototype for future AI‑driven infrastructure resilience.

