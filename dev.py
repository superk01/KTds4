import os
import json
import uuid
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
from langchain_openai import AzureOpenAIEmbeddings
from sklearn.metrics.pairwise import cosine_similarity
from langchain_core.messages import get_buffer_string

# ========== 환경 변수 및 초기 설정 ==========
load_dotenv()
HISTORY_DIR = "chat_history"
os.makedirs(HISTORY_DIR, exist_ok=True)

# UUID를 획득하여 세션아이디로 사용
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())  # 고유 UUID 생성

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

# ========== LLM 및 프롬프트 ==========
llm = AzureChatOpenAI(model=os.getenv("AZURE_OPENAI_LLM2"), temperature=0)

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
    # st.write(f"🔍 previous_question 결과: {previous_question}")
    # st.write(f"🔍 current_question 결과: {current_question}")
    # st.write(f"🔍 sim 결과: {sim}")
    return sim >= threshold

# ========== 검색 쿼리 생성 ==========
def generate_search_query(chat_history, latest_question):
    # 이전 질문과 의미적으로 연속인지 판단
    recent_user_questions = [msg["content"] for msg in chat_history if msg["role"] == "user"]
    # st.write(f"🔍 recent_user_questions 결과: {recent_user_questions}")
    if len(recent_user_questions) >= 1:
        previous_question = recent_user_questions[-1]
        #st.write(f"🔍 previous_question 결과: {previous_question}")
        if not is_follow_up_question(previous_question, latest_question):
            return latest_question  # 새로운 주제이므로 단독 사용

    # 전체 대화문 요약을 활용
    full_context = "\n".join(f"{msg['role'].capitalize()}: {msg['content']}" for msg in chat_history[-8:])
    # memory_context = get_buffer_string(memory.chat_memory.messages[-8:])

    # 질문 + 답변 포함한 맥락으로 쿼리 생성
    # combined_context = []
    # for msg in chat_history[-4:]:
    #     combined_context.append(f"{msg['role'].capitalize()}: {msg['content']}")

    prompt = ChatPromptTemplate.from_template(
        "다음은 사용자와 AI의 대화 기록입니다:\n{context}\n\n"
        "현재 질문: {question}\n\n"
        "위 대화를 기반으로 검색 엔진에 적합한 검색 쿼리를 생성해줘."
    )
    chain = prompt | llm | StrOutputParser()
    return chain.invoke({"context": full_context, "question": latest_question})


# def generate_search_query(latest_question):
#     memory_context = get_buffer_string(memory.chat_memory.messages[-8:])
    
#     prompt = ChatPromptTemplate.from_template(
#         """다음은 사용자와 AI의 대화 기록입니다:\n{context}\n\n
#         현재 질문: {question}\n\n
#         위 대화를 기반으로 검색 엔진에 적합한 검색 쿼리를 생성해줘."""
#     )
#     chain = prompt | llm | StrOutputParser()
#     return chain.invoke({
#         "context": memory_context,
#         "question": latest_question
#     })

def rephrase_question(latest_question):
    memory_context = get_buffer_string(memory.chat_memory.messages[-8:])
    prompt = ChatPromptTemplate.from_template(
        """이전 대화:\n{context}\n\n
        현재 질문: {question}\n\n
        문맥을 반영해 명확한 질문으로 바꿔줘."""
    )
    chain = prompt | llm | StrOutputParser()
    return chain.invoke({
        "context": memory_context,
        "question": latest_question
    })
# ========== 도구 정의 ==========
@tool
def web_search(query: str) -> str:
    """Search the web using Tavily."""
    tavily = TavilySearchResults(max_results=3, search_depth="advanced", include_answer=True, include_raw_content=True, include_images=True)
    # refined = rephrase_question(query)
    # search_query = generate_search_query( refined)

    # print("사용자가 입력한 query:", query)
    # print("최종 생성된 search_query:", search_query)

    # st.sidebar.write(f"입력된 query: {query}")
    # st.sidebar.write(f"생성된 search_query: {search_query}")

    # st.info(f"입력된 web_query: {search_query}")
    st.write(f"web_search")
    return tavily.invoke(query)

@tool
def law_search(query: str) -> str:
    """Search for Law information in PDF files."""
    retriever = AzureAISearchRetriever(content_key="chunk", top_k=3, index_name=os.getenv("AZURE_AI_SEARCH_INDEX_NAME_LAW"))
    llm_local = AzureChatOpenAI(deployment_name=os.getenv("AZURE_OPENAI_LLM1"))
    prompt = ChatPromptTemplate.from_template("""Answer the question based only on the context provided.\nContext: {context}\nQuestion: {question}""")
    chain = ({"context": retriever | format_docs, "question": RunnablePassthrough()} | prompt | llm_local | StrOutputParser())
    # refined = rephrase_question(query)
    # search_query = generate_search_query(refined)

    # print("사용자가 입력한 query:", query)
    # print("최종 생성된 search_query:", search_query)

    # st.sidebar.write(f"입력된 query: {query}")
    # st.sidebar.write(f"생성된 search_query: {search_query}")

    # st.info(f"입력된 law_query: {search_query}")
    st.write(f"law_search")
    return chain.invoke({"question": query})

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

prompt = ChatPromptTemplate.from_messages([
    # ("system", "당신은 지능형 AI Assistant입니다. 이전 질문과 맥락을 반영해 PDF 및 웹 검색 도구를 통해 질문에 답변합니다."),
#    ("system", "- 'law_search': PDF 법률 인덱스 검색\n- 'web_search': 최신 정보를 Tavily를 통해 웹에서 검색합니다."),
    # ("system", "세법 정보 검색은 PDF 파일에서 호텔 관련 정보를 찾고, 웹 검색은 Tavily를 통해 웹에서 정보를 찾습니다."),
    # ("system", "항상 친절하고 명확하게 응답하며, 불명확한 경우 확인을 요청합니다."),
    ("system", "너는 도구를 사용할 수 있는 AI Assistant야. 이전 대화 흐름을 반영하고 필요하면 도구를 사용해."),
    MessagesPlaceholder(variable_name="chat_history"),
    ("user", "{input}"),
    MessagesPlaceholder(variable_name="agent_scratchpad")
])

# ========== 에이전트 및 실행기 ==========
tools = [web_search, law_search]
agent = create_tool_calling_agent(llm, tools, prompt)
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
agent_executor = AgentExecutor(agent=agent, tools=tools, memory=memory, verbose=True)

# ========== Streamlit UI ==========
st.set_page_config(page_title="LAW & Web Chat Agent", layout="centered")
st.title("AI Assistant: 법률 & 웹 검색")

# user_id = st.text_input("사용자 이름 또는 ID를 입력하세요", key="user_id")
# if not user_id:
#     st.warning("대화 기록을 저장하려면 사용자 ID를 입력하세요.")
#     st.stop()

session_id = st.session_state.session_id

if "chat_history" not in st.session_state:
    st.session_state.chat_history = load_history(session_id)

if st.sidebar.button("🗑️ 대화 초기화"):
    st.session_state.chat_history = []
    save_history(session_id, [])
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
    memory.chat_memory.add_user_message(user_input)
    st.chat_message("user").write(user_input)

    #rephrase_question()

    # 2. 사용자 입력을 바탕으로 검색 쿼리 생성  
    rephrase_input = generate_search_query(st.session_state.chat_history, user_input)
    # st.write(f"🔍 최종 검색 쿼리: {rephrase_input}")
    
    with st.spinner("답변 생성 중..."):
        response = agent_executor.invoke({"input": rephrase_input})
        # 3. LangChain memory에서 최신 assistant 응답 가져오기
        assistant_reply = memory.chat_memory.messages[-1].content

        # 4. 응답 가져오기
        assistant_reply = response.get("output", str(response))
        
        # 5. 응답을 memory와 state에 추가
        memory.chat_memory.add_ai_message(assistant_reply)

        # st.write(f"🔍 메모리 결과: {memory.chat_memory.messages}")
        # st.write(f"🔍 history 결과: {st.session_state.chat_history}")
        #ontent = response.get("output", str(response))
        st.session_state.chat_history.append({"role": "assistant", "content": assistant_reply})
        st.chat_message("assistant").write(assistant_reply)
        save_history(session_id, st.session_state.chat_history)