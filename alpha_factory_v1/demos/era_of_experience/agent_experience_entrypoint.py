import os, asyncio, json, random, datetime as dt
from alpha_factory.core import AgentOrchestrator, StructuredToolCall
from openai_agents import Agent, OpenAIAgent, Tool
from reward_backends.fitness_reward import fitness_reward
from reward_backends.education_reward import education_reward

#---------------- LIFELONG EXPERIENCE STREAM ----------------------------------
def experience_stream():
    uid = 0
    while True:
        uid += 1
        ts  = dt.datetime.utcnow().isoformat()
        yield {
            "id":   uid,
            "time": ts,
            "user": random.choice(["alice", "bob"]),
            "context": random.choice([
                "morning‑run ✅ 5 km",
                "slept 6 h 20 m 😴",
                "completed Duolingo Spanish – 10 min",
                "requested dinner recipe low‑carb"
            ])
        }

#---------------- SENSOR‑MOTOR TOOLS ------------------------------------------
@Tool(name="web_search", description="query the internet")
async def web_search(q:str):
    return {"top_result": f"https://duckduckgo.com/?q={q.replace(' ','+')}"}

@Tool(name="plan_meal", description="craft a low‑carb meal plan")
async def plan_meal(cal:int=600):            # motor action stub
    return {"menu": f"Grilled salmon · greens · {cal} kcal"}

TOOLS = [web_search, plan_meal]

#---------------- ORCHESTRATION -----------------------------------------------
llm = OpenAIAgent(
    model  = os.getenv("MODEL_NAME", "gpt-4o-mini"),
    api_key= os.getenv("OPENAI_API_KEY", None),
    base_url=("http://ollama:11434/v1" if not os.getenv("OPENAI_API_KEY") else None)
)

agent = Agent(
    llm          = llm,
    memory       = "vector",                 # episodic memory pillar
    reward_fn    = lambda s,a,r: (
        0.5*fitness_reward(s,a,r) + 0.5*education_reward(s,a,r)
    ),
    tools        = TOOLS,
    planning     = "mcts",                   # non‑human planning
    name         = "Era‑Of‑Experience‑Agent"
)

orch = AgentOrchestrator(agent, experience_stream())
asyncio.run(orch.launch_gradio(port=7860))
