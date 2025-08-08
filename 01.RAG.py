from dotenv import load_dotenv
from langchain_openai import AzureChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain_community.retrievers import AzureAISearchRetriever
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
import os

load_dotenv()

llm2_name = os.getenv("AZURE_OPENAI_LLM2")
search_index_name = os.getenv("AZURE_AI_SEARCH_INDEX_NAME")


retriever = AzureAISearchRetriever(
                index_name=search_index_name,
                top_k=3,
                content_key="chunk"            
            )

#langchain의 공통함수 invoke를 활용하여 debugging
print(retriever.invoke("london hotel"))  # 검색 결과 확인

# 검색 결과를 하나의 text로 만듬
def format_docs(docs):
    return "\n\n".join([doc.page_content for doc in docs])

prompt = ChatPromptTemplate.from_template(
    """Answer the question based only on the context provided.".
        Context: {context}
        Question: {question}
    """
)

llm = AzureChatOpenAI(deployment_name=llm2_name)

chain = (
    {
        "context": retriever | format_docs,
        "question": RunnablePassthrough()
    }
    | prompt
    | llm
    | StrOutputParser()
    
)

result = chain.invoke("recommend a hotel in london")
print(result)