readme


1. 프로젝트 개요
  법률 관련 PDF 문서와 웹 검색을 통합하여 사용자 질문에 답변하는 AI 챗봇 개발
2. 기술 스택
  Python, Streamlit (웹 UI)
  LangChain (LLM 프롬프트, 에이전트, 메모리 관리)
  Azure OpenAI (LLM, 임베딩)
  Tavily (웹 검색 도구)
  Azure AI Search (법률 문서 검색)
  sklearn (코사인 유사도 계산)
4. 주요 기능
  3.1 세션 및 대화 히스토리 관리
      사용자별 고유 세션 ID(UUID) 생성 및 관리
      대화 기록을 JSON 파일로 저장/불러오기 (chat_history 폴더)
      Streamlit 세션 상태에 대화 히스토리 유지
  3.2 연속 질문 판단 (임베딩 기반)
      Azure OpenAI 임베딩을 사용해 이전 질문과 현재 질문의 의미적 유사도 계산
      코사인 유사도 기준(0.75 이상)으로 연속 질문 여부 판단
      연속 질문이면 이전 대화 맥락을 반영해 검색 쿼리 생성
  3.3 검색 쿼리 생성
      대화 최근 8개 메시지를 요약해 LLM에 전달
      LLM이 대화 맥락과 현재 질문을 바탕으로 검색에 적합한 쿼리 생성
      연속 질문이 아닐 경우, 단순히 최신 질문을 쿼리로 사용
5. 시연
   https://yh-webapp-011-a9haceb3aegfevh5.westus-01.azurewebsites.net/
6. 향후 개선 방향
   연속 질문 판단 로직 개선
   
