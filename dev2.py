import os
import json
import streamlit as st
from dotenv import load_dotenv
from langchain_community.retrievers import AzureAISearchRetriever
from langchain_openai import AzureChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain.memory import ConversationBufferMemory
from langchain_core.tools import tool
from langchain_community.tools import TavilySearchResults
from langchain.embeddings import AzureOpenAIEmbeddings
from sklearn.metrics.pairwise import cosine_similarity

# ========== 환경 변수 및 초기 설정 ==========
load_dotenv()
HISTORY_DIR = "chat_history"
os.makedirs(HISTORY_DIR, exist_ok=True)

# ========== 히스토리 관련 함수 ==========
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

# ========== 연속 질문 판단 함수 (임베딩 기반) ==========
embeddings = AzureOpenAIEmbeddings(
    azure_deployment=os.getenv("AZURE_OPENAI_EMBEDDING"),
    openai_api_version="2024-12-01-preview",
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_key=os.getenv("OPENAI_API_KEY"),
)

def is_follow_up_question(previous_question: str, current_question: str, threshold: float = 0.75) -> bool:
    prev_vec = embeddings.embed_query(previous_question)
    curr_vec = embeddings.embed_query(current_question)
    sim = cosine_similarity([prev_vec], [curr_vec])[0][0]
    return sim >= threshold

# ========== 검색 쿼리 생성 ==========
def generate_search_query(chat_history, latest_question):
    # 이전 질문과 의미적으로 연속인지 판단
    recent_user_questions = [msg["content"] for msg in chat_history if msg["role"] == "user"]
    if len(recent_user_questions) >= 1:
        previous_question = recent_user_questions[-1]
        if not is_follow_up_question(previous_question, latest_question):
            return latest_question  # 새로운 주제이므로 단독 사용

    # 질문 + 답변 포함한 맥락으로 쿼리 생성
    combined_context = []
    for msg in chat_history[-4:]:
        combined_context.append(f"{msg['role'].capitalize()}: {msg['content']}")

    prompt = ChatPromptTemplate.from_template(
        "이전 대화:\n{context}\n\n현재 질문: {question}\n\n"
        "위 대화를 참고해서 검색 엔진에 적합한 구체적인 검색 쿼리를 생성해줘."
    )
    chain = prompt | llm | StrOutputParser()
    return chain.invoke({"context": "\n".join(combined_context), "question": latest_question})

# ========== 도구 정의 ==========
@tool
def web_search(query: str) -> str:
    tavily = TavilySearchResults(max_results=3, search_depth="advanced", include_answer=True, include_raw_content=True, include_images=True)
    search_query = generate_search_query(st.session_state.chat_history, query)
    return tavily.invoke(search_query)

@tool
def law_search(query: str) -> str:
    retriever = AzureAISearchRetriever(content_key="chunk", top_k=3, index_name=os.getenv("AZURE_AI_SEARCH_INDEX_NAME_LAW"))
    llm_local = AzureChatOpenAI(deployment_name=os.getenv("AZURE_OPENAI_LLM1"))
    prompt = ChatPromptTemplate.from_template("""Answer the question based only on the context provided.\nContext: {context}\nQuestion: {question}""")
    chain = ({"context": retriever | format_docs, "question": RunnablePassthrough()} | prompt | llm_local | StrOutputParser())
    search_query = generate_search_query(st.session_state.chat_history, query)
    return chain.invoke({"question": search_query})

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# ========== LLM 및 프롬프트 ==========
llm = AzureChatOpenAI(model=os.getenv("AZURE_OPENAI_LLM2"), temperature=0)

prompt = ChatPromptTemplate.from_messages([
    ("system", "당신은 지능형 AI Assistant입니다. 이전 질문과 맥락을 반영해 PDF 및 웹 검색 도구를 통해 질문에 답변합니다."),
    ("system", "- 'law_search': PDF 법률 인덱스 검색\n- 'web_search': 최신 정보를 Tavily를 통해 웹에서 검색합니다."),
    ("system", "항상 친절하고 명확하게 응답하며, 불명확한 경우 확인을 요청합니다."),
    ("user", "{input}"),
    MessagesPlaceholder(variable_name="agent_scratchpad")
])

# ========== 에이전트 및 실행기 ==========
tools = [web_search, law_search]
agent = create_tool_calling_agent(llm, tools, prompt)
memory = ConversationBufferMemory(return_messages=True)
agent_executor = AgentExecutor(agent=agent, tools=tools, memory=memory, verbose=True)

# ========== Streamlit UI ==========
st.set_page_config(page_title="Hotel & Web Chat Agent", layout="centered")
st.title("AI Assistant: 호텔 & 웹 검색")

user_id = st.text_input("사용자 이름 또는 ID를 입력하세요", key="user_id")
if not user_id:
    st.warning("대화 기록을 저장하려면 사용자 ID를 입력하세요.")
    st.stop()

if "chat_history" not in st.session_state:
    st.session_state.chat_history = load_history(user_id)

if st.sidebar.button("🗑️ 대화 초기화"):
    st.session_state.chat_history = []
    save_history(user_id, [])
    st.rerun()

# 사이드바에 대화 히스토리 요약 표시
st.sidebar.markdown("## 💬 이전 대화 보기")
chat_pairs, current_pair = [], {}
for msg in st.session_state.chat_history:
    if msg["role"] == "user":
        if current_pair and "user" in current_pair:
            chat_pairs.append(current_pair)
        current_pair = {"user": msg["content"], "assistant": ""}
    elif msg["role"] == "assistant" and current_pair:
        current_pair["assistant"] = msg["content"]
if current_pair and "user" in current_pair:
    chat_pairs.append(current_pair)

for i, pair in enumerate(chat_pairs):
    with st.sidebar.expander(f"❓ Q{i+1}: {pair['user'][:30]}..."):
        st.markdown(f"**👤 사용자:** {pair['user']}")
        st.markdown(f"**🤖 어시스턴트:** {pair['assistant']}")

# 메인 챗 메시지 출력
for msg in st.session_state.chat_history:
    st.chat_message(msg["role"]).write(msg["content"])

# 사용자 입력 받기
user_input = st.chat_input("질문을 입력하세요...")
if user_input:
    st.session_state.chat_history.append({"role": "user", "content": user_input})
    st.chat_message("user").write(user_input)
    with st.spinner("답변 생성 중..."):
        response = agent_executor.invoke({"input": user_input})
        content = response.get("output", str(response))
        st.session_state.chat_history.append({"role": "assistant", "content": content})
        st.chat_message("assistant").write(content)
        save_history(user_id, st.session_state.chat_history)
