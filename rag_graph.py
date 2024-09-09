from typing import Annotated
from typing_extensions import TypedDict, List
from langchain_core.messages import AIMessage
from langgraph.graph import START, END, StateGraph
from langgraph.graph.message import AnyMessage, add_messages
from langgraph.checkpoint.memory import MemorySaver

from utils import enter_chain

from documents import prepare_db_rag
from prompts import create_prompts_rag

retriever = prepare_db_rag()
rag_chain, regenerate_chain, retrieval_grader, hallucination_grader, answer_grader = create_prompts_rag()


class RAGState(TypedDict):
    """
    Represents the state of our graph.

    Attributes:
        question: question
        generation: LLM generation
        documents: list of documents
    """
    question: str
    generation: str
    regenerate: str
    documents: List[str]
    messages: Annotated[list[AnyMessage], add_messages]

### Nodes

def retrieve(state):
    """
    Retrieve documents

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): New key added to state, documents, that contains retrieved documents
    """
    question = state["messages"][-1].content
    documents = retriever.invoke(question)

    return {"documents": documents, "question": question}

def generate(state):
    """
    Generate answer

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): New key added to state, generation, that contains LLM generation
    """

    question = state["question"]
    documents = state["documents"]
    generation = rag_chain.invoke({"documents": documents, "question": question})
    return {
        "documents": documents,
        "question": question,
        "generation": generation,
    }

def grade_documents(state):
    """
    Determines whether the retrieved documents are relevant to the question.

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): Updates documents key with only filtered relevant documents
    """

    question = state["question"]
    documents = state["documents"]
    filtered_docs = []
    regenerate = "no"
    for d in documents:
        score = retrieval_grader.invoke(
            {"question": question, "document": d.page_content}
        )
        grade = score["score"]
        if grade == "yes":
            filtered_docs.append(d)
        else:
            regenerate = "yes"
            continue
    return {
        "documents": filtered_docs,
        "question": question,
        "regenerate": regenerate,
    }

def regenerate_question(state):
    
    question = state["question"]
    documents = state["documents"]
    regeneration = regenerate_chain.invoke({"documents": documents, "question": question})
    return {
        "documents": documents,
        "question": question,
        "generation": regeneration,
    }

def send_response(state):
    message = state["generation"]
    return {"messages": AIMessage(content=message)}

### Conditional Edges

def decide_to_generate(state):
    """
    Determines whether to generate an answer, or re-generate a question.

    Args:
        state (dict): The current graph state

    Returns:
        str: Binary decision for next node to call
    """
    regenerate = state["regenerate"]
    if regenerate == "yes":
        return "regenerate"
    else:
        return "generate"

def grade_generation_v_documents_and_question(state):
    """
    Determines whether the generation is grounded in the document and answers question.

    Args:
        state (dict): The current graph state

    Returns:
        str: Decision for next node to call
    """

    question = state["question"]
    documents = state["documents"]
    generation = state["generation"]

    score = hallucination_grader.invoke(
        {"documents": documents, "generation": generation}
    )
    grade = score["score"]

    # Check hallucination
    if grade == "yes":
        # Check question-answering
        score = answer_grader.invoke({"question": question, "generation": generation})
        grade = score["score"]
        if grade == "yes":
            return "useful"
        else:
            return "not useful"
    else:
        return "not supported"

# Graph

def rag_compile():
    workflow = StateGraph(RAGState)

    # Define the nodes
    workflow.add_node("retrieve", retrieve)  # retrieve
    workflow.add_node("grade_documents", grade_documents)  # grade documents
    workflow.add_node("generate", generate)  # generate
    workflow.add_node("regenerate_question", regenerate_question)  # regenerate_question
    workflow.add_node("send_response", send_response)  # Send final answer

    # Build graph
    workflow.add_edge(START, "retrieve")
    workflow.add_edge("retrieve", "grade_documents")
    workflow.add_conditional_edges(
        "grade_documents",
        decide_to_generate,
        {
            "regenerate": "regenerate_question",
            "generate": "generate",
        },
    )
    workflow.add_conditional_edges(
        "generate",
        grade_generation_v_documents_and_question,
        {
            "not supported": "generate",
            "useful": "send_response",
            "not useful": "regenerate_question",
        },
    )

    workflow.add_edge("regenerate_question", "send_response")
    workflow.add_edge("send_response", END)


    memory = MemorySaver() #! Only for debugging and testing, check others
    rag_graph = workflow.compile(checkpointer=memory)

    rag_work_chain = enter_chain | rag_graph
    return rag_work_chain, rag_graph

_ , model = rag_compile()