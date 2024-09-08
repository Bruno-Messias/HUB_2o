from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from langchain_groq import ChatGroq


def create_prompts_rag():

    llm = ChatGroq(
        model="llama-3.1-8b-instant",
        temperature=0,
    )

    prompt = PromptTemplate(
        template="""You are an assistant for question-answering tasks.

        Use the following documents to answer the question.

        If you don't know the answer, just say that you don't know.
        All the answer must be formatted as Markdown to the user.

        Use three sentences maximum and keep the answer concise:
        Question: {question}
        Documents: {documents}
        Answer:
        """,
        input_variables=["question", "documents"],
    )

    rag_chain = prompt | llm | StrOutputParser()


    prompt = PromptTemplate(
        template="""You are an assistant for question-answering tasks.
        
        Tell to the user that you don't know the answer for his question 
        with the documents you have access, 
        or you need more explanation of what he want.
        
        Use three sentences maximum and keep the answer concise:
        Question: {question}
        Documents: {documents}
        Answer:
        """,
        input_variables=["question", "documents"],
    )

    regenerate_chain = prompt | llm | StrOutputParser()

    # JSON
    prompt = PromptTemplate(
        template="""You are a grader assessing relevance of a retrieved document to a user question. \n
        Here is the retrieved document: \n\n {document} \n\n
        Here is the user question: {question} \n
        If the document contains keywords related to the user question, grade it as relevant. \n
        It does not need to be a stringent test. The goal is to filter out erroneous retrievals. \n
        Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question. \n
        Provide the binary score as a JSON with a single key 'score' and no premable or explanation.""",
        input_variables=["question", "document"],
    )

    retrieval_grader = prompt | llm | JsonOutputParser()

    prompt = PromptTemplate(
        template="""You are a grader assessing whether 
        an answer is grounded in / supported by a set of facts. Give a binary 'yes' or 'no' score to indicate 
        whether the answer is grounded in / supported by a set of facts. Provide the binary score as a JSON with a 
        single key 'score' and no preamble or explanation.
        Here are the facts:
        \n ------- \n
        {documents} 
        \n ------- \n
        Here is the answer: {generation} """,
        input_variables=["generation", "documents"],
    )

    hallucination_grader = prompt | llm | JsonOutputParser()

    prompt = PromptTemplate(
        template="""You are a grader assessing whether an answer is useful to resolve a question. 
        Give a binary score 'yes' or 'no' to indicate whether the answer is useful to resolve a question. 
        Provide the binary score as a JSON with a single key 'score' and no preamble or explanation.
        Here is the answer:
        \n ------- \n
        {generation} 
        \n ------- \n
        Here is the question: {question}""",
        input_variables=["generation", "question"],
    )

    answer_grader = prompt | llm | JsonOutputParser()
    return rag_chain, regenerate_chain, retrieval_grader, hallucination_grader, answer_grader