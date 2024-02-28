from langchain.prompts.prompt import PromptTemplate
from langchain_core.prompts.chat import ChatPromptTemplate

def load_prompts():
    # ### Create PromptTemplate and LLMChain

    _template = """
    [INST] 
    Given the following conversation and a follow up question, rephrase the follow up question to be a standalone question, in its original language, that can be used to query a FAISS index. This query will be used to retrieve documents with additional context. 

    Let me share a couple examples that will be important. 

    If you do not see any chat history, you MUST return the "Follow Up Input" as is:

    ```
    Chat History:

    Follow Up Input: How is Lawrence doing?
    Standalone Question:
    How is Lawrence doing?
    ```

    If this is the second question onwards, you should properly rephrase the question like this:

    ```
    Chat History:
    Human: How is Lawrence doing?
    AI: 
    Lawrence is injured and out for the season.

    Follow Up Input: What was his injurt?
    Standalone Question:
    What was Lawrence's injury?
    ```

    Now, with those examples, here is the actual chat history and input question.

    Chat History:
    {chat_history}

    Follow Up Input: {question}
    Standalone question:
    [your response here]
    [/INST] 
    """
    CONDENSE_QUESTION_PROMPT = PromptTemplate.from_template(_template)


    # In[14]:


    template = """
    [INST] 
    Answer the question based only on the following context:
    {context}

    Question: {question}
    [/INST] 
    """
    ANSWER_PROMPT = ChatPromptTemplate.from_template(template)

    DEFAULT_DOCUMENT_PROMPT = PromptTemplate.from_template(template="{page_content}")

    return CONDENSE_QUESTION_PROMPT, ANSWER_PROMPT, DEFAULT_DOCUMENT_PROMPT
