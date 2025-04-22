"""
AI‑GA Demo – distils Clune (2020) three‑pillar paradigm. :contentReference[oaicite:0]{index=0}&#8203;:contentReference[oaicite:1]{index=1}
• Pillar 1  meta‑learning architectures   → simple NAS via mutation
• Pillar 2  meta‑learning algorithms      → inner‑loop SGD vs Hebbian flag
• Pillar 3  learning‑environment creator  → self‑synthesised curriculum
All visualised in a Gradio dashboard (port 7862).
"""
import os, asyncio
from openai_agents import Agent, OpenAIAgent, Tool
from meta_evolver import MetaEvolver
from curriculum_env import CurriculumEnv
import gradio as gr

LLM = OpenAIAgent(
    model=os.getenv("MODEL_NAME", "gpt-4o-mini"),
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url=("http://ollama:11434/v1" if not os.getenv("OPENAI_API_KEY") else None)
)

@Tool(name="describe_candidate", description="explain current agent design")
async def describe_candidate(arch:str):
    return LLM(f"In 2 sentences, describe why this architecture '{arch}' might learn fast.")

async def launch():
    evo = MetaEvolver(env_cls=CurriculumEnv, llm=LLM)
    with gr.Blocks(title="AI‑GA Meta‑Evolution Demo") as ui:
        plot = gr.LinePlot(label="Fitness by Generation")
        log  = gr.Markdown()
        def step(gens:int=5):
            evo.run_generations(gens)
            return evo.history_plot(), evo.latest_log()
        gr.Button("Evolve 5 Generations").click(step, inputs=[], outputs=[plot, log])
    ui.launch(server_name="0.0.0.0", server_port=7862)

if __name__ == "__main__":
    asyncio.run(launch())
