
from fastapi import FastAPI, Body
import uvicorn, os
from pydantic import BaseModel
app=FastAPI()

class Task(BaseModel):
    agent:str
    payload:dict

@app.post('/rpc')
def rpc(task:Task):
    # stub: acknowledge
    return {'status':'accepted','agent':task.agent}

if __name__=='__main__':
    uvicorn.run(app,host='0.0.0.0',port=int(os.getenv('RPC_PORT','8000')))
