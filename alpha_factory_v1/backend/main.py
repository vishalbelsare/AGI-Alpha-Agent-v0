
from .orchestrator import Orchestrator
def main():
    orch=Orchestrator()
    orch.run_forever()
if __name__=='__main__':
    main()
