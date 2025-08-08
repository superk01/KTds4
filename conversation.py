from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain_community.tools import TavilySearchResults
from langchain.memory import ConversationBufferMemory
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

# 1. Tavily 도구 정의
tavily_tool = TavilySearchResults(max_results=3)

# 2. LLM 설정
llm = ChatOpenAI(model="gpt-4o", temperature=0)  # 또는 gpt-3.5-turbo

# 3. 프롬프트 정의
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant that answers questions using tools."),
    ("user", "{input}"),
    MessagesPlaceholder(variable_name="agent_scratchpad")
])

# 4. 메모리 설정 (대화 맥락 저장)
memory = ConversationBufferMemory(return_messages=True)

# 5. 에이전트 & 실행기
agent = create_tool_calling_agent(llm, [tavily_tool], prompt)

agent_executor = AgentExecutor(
    agent=agent,
    tools=[tavily_tool],
    memory=memory,
    verbose=True
)

# 1차 질문
response1 = agent_executor.invoke({"input": "1~10까지 정수는 뭐야"})
print("응답1:", response1["output"])

# 2차 질문 (이전 답변을 바탕으로 추가 질문)
response2 = agent_executor.invoke({"input": "그 중 짝수는 뭐야?"})
print("응답2:", response2["output"])
