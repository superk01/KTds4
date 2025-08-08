import os
from langchain_community.retrievers import AzureAISearchRetriever
from langchain_openai import AzureChatOpenAI
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_community.tools import TavilySearchResults
from langchain.agents import create_tool_calling_agent
from langchain.agents import AgentExecutor
from langchain_core.tools import tool
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.memory import ConversationBufferMemory
import streamlit as st
import json

load_dotenv()

#from langchain.agents import AgentExecutor

# 기본 경로 설정
HISTORY_DIR = "chat_history"
os.makedirs(HISTORY_DIR, exist_ok=True)

# 🔹 히스토리 저장 및 불러오기
def get_history_path(user_id):
    return os.path.join(HISTORY_DIR, f"{user_id}.json")

def save_history(user_id, history):
    with open(get_history_path(user_id), "w", encoding="utf-8") as f:
        json.dump(history, f, ensure_ascii=False, indent=2)

def load_history(user_id):
    path = get_history_path(user_id)
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return []

def generate_search_query(chat_history, latest_question):
    # 직전 사용자 메시지만 가져오기 (또는 전체 대화 요약도 가능)
    context = [msg["content"] for msg in chat_history if msg["role"] == "user"][-2:]

    prompt = ChatPromptTemplate.from_template(
        "이전 질문들: {context}\n\n현재 질문: {question}\n\n위 맥락을 반영해서 검색 엔진에 적절한 검색 쿼리를 생성해줘."
    )

    chain = prompt | llm | StrOutputParser()

    result = chain.invoke({
        "context": "\n".join(context),
        "question": latest_question
    })

    return result

user_id = st.text_input("사용자 이름 또는 ID를 입력하세요", key="user_id")

if not user_id:
    st.warning("대화 기록을 저장하려면 사용자 ID를 입력하세요.")
    st.stop()

@tool
def web_search(query: str) -> str:   
    """Search the web using Tavily."""

    tavilyRetriever = TavilySearchResults(
        max_results=3,  # 반환할 결과의 수
        search_depth="advanced",  # 검색 깊이: basic 또는 advanced
        include_answer=True,  # 결과에 직접적인 답변 포함
        include_raw_content=True,  # 페이지의 원시 콘텐츠 포함
        include_images=True,  # 결과에 이미지 포함
        # include_domains=[...],  # 특정 도메인으로 검색 제한
        # exclude_domains=[...],  # 특정 도메인 제외
        # name="...",  # 기본 도구 이름 덮어쓰기
        # description="...",  # 기본 도구 설명 덮어쓰기
        # args_schema=...,  # 기본 args_schema 덮어쓰기
    )

    # 이전 대화 이력 로깅
    print(json.dumps(st.session_state.chat_history, ensure_ascii=False, indent=2))
    
    # 최신 대화와 연결 (필요시 chat_history 를 받도록 수정)
    query_to_search = generate_search_query(
        st.session_state.chat_history,  # 대화 이력
        query                           # 현재 질문
    )
    
    return tavilyRetriever.invoke(query_to_search)

@tool
def hotel_search(query: str) -> str:
    """Search for Hotel information in PDF files."""
    retriever = AzureAISearchRetriever(
        content_key="chunk",      # 인덱스 내 컨텐츠 필드명
        top_k=3,                   # 반환할 검색 결과 개수
        index_name=os.getenv("AZURE_AI_SEARCH_INDEX_NAME"),  # Azure Search 인덱스 이름
    )

    llm = AzureChatOpenAI(
        deployment_name=os.getenv("AZURE_OPENAI_LLM1"),  # 환경 변수명 확인
    )

    prompt = ChatPromptTemplate.from_template(
        """Answer the question based only on the context provided.
        Context: {context}
        Question: {question}"""
    )

    chain = (
        {
            "context": retriever | format_docs,
            "question": RunnablePassthrough()
        }
        | prompt
        | llm
        | StrOutputParser()
    )

    # 최신 대화와 연결 (필요시 chat_history 를 받도록 수정)
    query_to_search = generate_search_query(
        st.session_state.chat_history,  # 대화 이력
        query                           # 현재 질문
    )
    # 체인 실행
    result = chain.invoke({"question": query_to_search})
    return result

@tool
def law_search(query: str) -> str:
    """Search for Law information in PDF files."""
    retriever = AzureAISearchRetriever(
        content_key="chunk",      # 인덱스 내 컨텐츠 필드명
        top_k=3,                   # 반환할 검색 결과 개수
        index_name=os.getenv("AZURE_AI_SEARCH_INDEX_NAME_LAW"),  # Azure Search 인덱스 이름
    )

    llm = AzureChatOpenAI(
        deployment_name=os.getenv("AZURE_OPENAI_LLM1"),  # 환경 변수명 확인
    )

    prompt = ChatPromptTemplate.from_template(
        """Answer the question based only on the context provided.
        Context: {context}
        Question: {question}"""
    )

    chain = (
        {
            "context": retriever | format_docs,
            "question": RunnablePassthrough()
        }
        | prompt
        | llm
        | StrOutputParser()
    )

    # 이전 대화 이력 로깅
    print(json.dumps(st.session_state.chat_history, ensure_ascii=False, indent=2))

    # 최신 대화와 연결 (필요시 chat_history 를 받도록 수정)
    query_to_search = generate_search_query(
        st.session_state.chat_history,  # 대화 이력
        query                           # 현재 질문
    )

    # 체인 실행
    result = chain.invoke({"question": query_to_search})
    return result

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


#tools = [web_search, hotel_search]  # 사용할 도구 목록
tools = [web_search, law_search]  # 사용할 도구 목록


llm = AzureChatOpenAI(model=os.getenv("AZURE_OPENAI_LLM2"),temperature=0)

# 프롬프트 템플릿 정의
# prompt = ChatPromptTemplate.from_messages([
#     ("system", "당신은 사용자의 요청을 처리하는 AI Assistant입니다. 기본적으로는 호텔 정보 검색을 담당하며, 웹 검색 도구를 통해 추가 정보를 제공합니다."),
#     ("system", "사용자의 질문을 먼저 영어로 번역하고 이해합니다."),    
#     ("system", "질문에 따라 두 가지 도구를 사용할 수 있습니다: 호텔 정보 검색과 웹 검색."),
#     ("system", "호텔 정보 검색은 PDF 파일에서 호텔 관련 정보를 찾고, 웹 검색은 Tavily를 통해 웹에서 정보를 찾습니다."),
#     ("user", "{input}"),
#     MessagesPlaceholder(variable_name="agent_scratchpad")
# ])

prompt = ChatPromptTemplate.from_messages([
    ("system", 
     "당신은 사용자의 요청을 처리하는 지능형 AI Assistant입니다. "
     "기본적으로 질문에 대한 답은 PDF 파일에서 검색하며, 필요할 경우 웹 검색 도구를 통해 최신 정보를 보완합니다."),

    # ("system", 
    #  "입력된 사용자의 질문이 한글일 경우, 내부적으로 영어로 번역하여 처리한 후 적절한 도구를 선택해 응답을 생성합니다."),
    ("system","질문을 받은 후 사용자의 질문이 이전 질문과 연관되어 있는지 확인하고, 이전 질문과 관련된 정보가 있다면 그 정보를 활용하여 답변을 생성합니다."),

    ("system", 
     "사용자의 질문 유형에 따라 다음 두 가지 도구 중 하나를 사용할 수 있습니다:\n"
     "- 'law_search': Azure Cognitive Search를 통해 사전에 인덱싱된 PDF에서 법률 정보를 검색합니다.\n"
     "- 'web_search': law_search로 찾지 못한 내용을 Tavily 검색 API를 사용하여 웹에서 최신 정보나 추가 정보를 수집합니다."),

    ("system", 
     "모든 응답은 친절하고, 명확하며, 사용자의 질문에 직접적으로 관련된 정보만 포함해야 합니다. "
     "불필요한 추측은 피하고, 명확하지 않은 경우 확인을 요청하세요."),

    ("user", "{input}"),
    MessagesPlaceholder(variable_name="agent_scratchpad")
])


# 에이전트 생성 (도구 호출)
agent = create_tool_calling_agent(llm, tools, prompt)

memory = ConversationBufferMemory(return_messages=True)

# 에이전트 실행기 생성
agent_executor = AgentExecutor(
    agent=agent,      # 도구 호출 에이전트
    tools=tools,      # 도구 목록
    memory=memory,    # 대화 메모리
    verbose=True      # 상세 로그 출력
    )

# Streamlit UI 설정
st.set_page_config(page_title="Hotel & Web Chat Agent", layout="centered")
st.title("AI Assistant: 호텔 & 웹 검색")

# 🔹 대화 초기화
if st.sidebar.button("🗑️ 대화 초기화"):
    st.session_state.chat_history = []
    save_history(user_id, [])
    st.rerun()

if "chat_history" not in st.session_state:
#    st.session_state.chat_history = []
    st.session_state.chat_history = save_history(user_id)

# 🔹 사이드바 대화 리스트 UI
st.sidebar.markdown("## 💬 이전 대화 보기")

# 대화를 질문-답변 단위로 그룹핑
chat_pairs = []
current_pair = {}
for msg in st.session_state.chat_history:
    if msg["role"] == "user":
        if current_pair and "user" in current_pair:
            chat_pairs.append(current_pair)
        current_pair = {"user": msg["content"], "assistant": ""}
    elif msg["role"] == "assistant" and current_pair:
        current_pair["assistant"] = msg["content"]
# 마지막 질문-답변 쌍 추가
if current_pair and "user" in current_pair:        
    chat_pairs.append(current_pair)

# 사이드바에 요약 표시
for i, pair in enumerate(chat_pairs):
    with st.sidebar.expander(f"❓ Q{i+1}: {pair['user'][:30]}..."):
        st.markdown(f"**👤 사용자:** {pair['user']}")
        st.markdown(f"**🤖 어시스턴트:** {pair['assistant']}")

# 챗 메시지 출력
for msg in st.session_state.chat_history:
    st.chat_message(msg["role"]).write(msg["content"])

# 사용자 입력 받기
user_input = st.chat_input("질문을 입력하세요...")
if user_input:
    st.session_state.chat_history.append({"role": "user", "content": user_input})
    st.chat_message("user").write(user_input)
    with st.spinner("답변 생성 중..."):
#        response = agent_executor.invoke({"input": user_input, "chat_history": st.session_state.chat_history})
        response = agent_executor.invoke({"input": user_input})
        content = response.get("output", str(response))
        st.session_state.chat_history.append({"role": "assistant", "content": content})
        st.chat_message("assistant").write(content)
        load_history(user_id)



