from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
from langserve import add_routes
import uvicorn
import uuid

from system_graph import model

load_dotenv()

app = FastAPI(
    title="LangServe AI Agent",
    version="1.0",
    description="LangGraph backend for the AI Agents Masterclass series agent.",
)

# Set all CORS enabled origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"],
)

def main():
    # Fetch the AI Agent LangGraph runnable which generates the workouts

    # Create the Fast API route to invoke the runnable
    # config = {"configurable": {"thread_id": str(uuid.uuid4()),
    #                         "passenger_id": "3442 587242",
    #                         },
    #         "recursion_limit": 20}
    
    # model = compile_model()

    # Edit this to add the chain you want to add
    add_routes(
        app, 
        model)
    # Start the API
    uvicorn.run(app, host="0.0.0.0", port=8000)

if __name__ == "__main__":
    main()



