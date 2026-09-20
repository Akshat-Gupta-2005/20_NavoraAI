from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from agent.agentic_workflow import GraphBuilder
from utils.save_to_document import save_document
from starlette.responses import JSONResponse
import os
import datetime
from dotenv import load_dotenv
from pydantic import BaseModel
load_dotenv()

# Built once on startup and reused for every request.
react_app = None


def _dump_graph_png(app_graph, filename: str = "my_graph.png") -> None:
    """Best-effort render of the graph diagram; never fatal."""
    try:
        png_graph = app_graph.get_graph().draw_mermaid_png()
        with open(filename, "wb") as f:
            f.write(png_graph)
        print(f"Graph saved as '{filename}' in {os.getcwd()}")
    except Exception as e:
        print(f"Skipped graph render: {e}")


@asynccontextmanager
async def lifespan(app: FastAPI):
    global react_app
    graph = GraphBuilder(model_provider="groq")
    react_app = graph()
    _dump_graph_png(react_app)
    yield


app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # set specific origins in prod
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
class QueryRequest(BaseModel):
    question: str

@app.post("/query")
async def query_travel_agent(query:QueryRequest):
    try:
        print(query)
        # Assuming request is a pydantic object like: {"question": "your text"}
        messages={"messages": [query.question]}
        output = react_app.invoke(messages)

        # If result is dict with messages:
        if isinstance(output, dict) and "messages" in output:
            final_output = output["messages"][-1].content  # Last AI response
        else:
            final_output = str(output)

        return {"answer": final_output}
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})
