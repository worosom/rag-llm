#!/usr/bin/env python
# coding: utf-8

# ### Dependencies
import sys
import os
import pickle
import time
import torch

from langchain_core.messages import get_buffer_string
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain.memory import ConversationBufferMemory

from operator import itemgetter

from llm import load_llms

from data import load_retriever, _combine_documents

from prompts import load_prompts



def load_chain(retriever, prompts):
    standalone_query_generation_llm, response_generation_llm = load_llms()

    CONDENSE_QUESTION_PROMPT, ANSWER_PROMPT, DEFAULT_DOCUMENT_PROMPT = prompts
    # Instantiate ConversationBufferMemory
    memory = ConversationBufferMemory(
     return_messages=True, output_key="answer", input_key="question"
    )

    # First we add a step to load memory
    # This adds a "memory" key to the input object
    loaded_memory = RunnablePassthrough.assign(
        chat_history=RunnableLambda(memory.load_memory_variables) | itemgetter("history"),
    )

    # Now we calculate the standalone question
    standalone_question = {
        "standalone_question": {
            "question": lambda x: x["question"],
            "chat_history": lambda x: get_buffer_string(x["chat_history"]),
        }
        | CONDENSE_QUESTION_PROMPT
        | standalone_query_generation_llm,
    }
    # Now we retrieve the documents
    retrieved_documents = {
        "docs": itemgetter("standalone_question") | retriever,
        "question": lambda x: x["standalone_question"],
    }
    # Now we construct the inputs for the final prompt
    final_inputs = {
        "context": lambda x: _combine_documents(x["docs"], DEFAULT_DOCUMENT_PROMPT),
        "question": itemgetter("question"),
    }

    # And finally, we do the part that returns the answers
    answer = {
        "answer": final_inputs | ANSWER_PROMPT | response_generation_llm,
        "question": itemgetter("question"),
        "context": final_inputs["context"]
    }

    # And now we put it all together!
    final_chain = loaded_memory | standalone_question | retrieved_documents | answer
    return final_chain, memory


def call_conversational_rag(question, chain, memory):
    """
    Calls a conversational RAG (Retrieval-Augmented Generation) model to generate an answer to a given question.

    This function sends a question to the RAG model, retrieves the answer, and stores the question-answer pair in memory 
    for context in future interactions.

    Parameters:
    question (str): The question to be answered by the RAG model.
    chain (LangChain object): An instance of LangChain which encapsulates the RAG model and its functionality.
    memory (Memory object): An object used for storing the context of the conversation.

    Returns:
    dict: A dictionary containing the generated answer from the RAG model.
    """

    # Prepare the input for the RAG model
    inputs = {"question": question}

    # Invoke the RAG model to get an answer
    result = chain.invoke(inputs)

    # Save the current question and its answer to memory for future context
    memory.save_context(inputs, {"answer": result["answer"]})

    # Return the result
    return result


if __name__ == '__main__':
    print("Loading prompts")
    prompts = load_prompts()
    CONDENSE_QUESTION_PROMPT, ANSWER_PROMPT, DEFAULT_DOCUMENT_PROMPT = prompts
    print("Creating retriever")
    retriever = load_retriever(sys.argv[1], DEFAULT_DOCUMENT_PROMPT)
    print("Creating chain")
    final_chain, memory = load_chain(retriever, prompts)
    print("Ready.")

    print("=" * 80)
    question = "Who studied yellow-legged frogs?"
    print("Question:", question)
    start = time.time()
    result = call_conversational_rag(question, final_chain, memory)
    end = time.time()
    print("=" * 40)
    print("Answer:", result["answer"])
    print("=" * 40)
    print("Context:", result["context"])
    print("=" * 40)
    print("Time taken: %.2f" % (end - start,))
    print("=" * 80)
    question = "What kind of indigenous people did he meet during his journey?"
    print("Question:", question)
    start = time.time()
    result = call_conversational_rag(question, final_chain, memory)
    end = time.time()
    print("=" * 40)
    print("Answer:", result["answer"])
    print("=" * 40)
    print("Context:", result["context"])
    print("=" * 40)
    print("Time taken: %.2f" % (end - start,))
    print("=" * 80)
