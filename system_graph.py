from typing import Annotated
from typing_extensions import TypedDict
from langchain_openai import ChatOpenAI
from langgraph.graph import START, END, StateGraph
from langgraph.graph.message import AnyMessage, add_messages
from langgraph.checkpoint.memory import MemorySaver

from utils import create_team_supervisor

from tools_graph import tools_compile
from rag_graph import rag_compile

class SystemState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    next: str

def get_last_message(state: SystemState) -> str:
    return state["messages"][-1].content


def join_graph(response: dict):
    return {"messages": [response["messages"][-1]]}


def compile_model():
    llm = ChatOpenAI(model="gpt-4o")

    supervisor_node = create_team_supervisor(
        llm,
        "You are a Assistant tasked with managing two separate workflows between the"
        " following assistants: {team_members}. Given the following user request,"
        " respond with the worker to act next. Each worker will perform a"
        " task and respond with their results and status."
        "Any Question about Flights, Cars, Hotel or Travel, select the ToolsWorkflow"
        "When finished, or the assistant can't respond the answer,"
        " respond with FINISH.",
        ["RAGWorkflow", "ToolsWorkflow"],
    )
    
    rag_work_chain = rag_compile()
    tools_work_chain = tools_compile()
    
    super_graph = StateGraph(SystemState)
    super_graph.add_node("RAGWorkflow", get_last_message | rag_work_chain | join_graph)
    super_graph.add_node("ToolsWorkflow", get_last_message | tools_work_chain | join_graph)
    super_graph.add_node("PrimaryAssistant", supervisor_node)

    super_graph.add_edge(START, "PrimaryAssistant")
    super_graph.add_conditional_edges(
        "PrimaryAssistant",
        lambda x: x["next"],
        {
            "ToolsWorkflow": "ToolsWorkflow",
            "RAGWorkflow": "RAGWorkflow",
            "FINISH": END,
        },
    )
    super_graph.add_edge("RAGWorkflow", END)
    super_graph.add_edge("ToolsWorkflow", END)

    memory = MemorySaver() #! Only for debugging and testing, check others
    super_graph = super_graph.compile(checkpointer=memory)
    return super_graph

model = compile_model()