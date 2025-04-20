
from .agent_base import AgentBase
import random, logging
class BiotechAgent(AgentBase):
    def observe(self):
        # placeholder: pretend we scanned new papers
        finding=f"gene_{random.randint(1,100)} breakthrough"
        self.memory.write(self.name,'observation',{'paper':finding})
        return [{'paper':finding}]
    def think(self,obs):
        idea={'type':'research','topic':obs[-1]['paper'],'impact':random.random()}
        self.memory.write(self.name,'idea',idea)
        return [idea]
    def act(self,tasks):
        for t in tasks:
            self.memory.write(self.name,'action',{'proposal':t,'status':'submitted'})
