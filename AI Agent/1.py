# Импорты
import os
import time
import warnings
from typing import List, Optional, TypedDict

# LangChain core
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.runnables import RunnableBranch, RunnableLambda, RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# LLM и эмбеддинги
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

# Инструменты и векторные БД
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter

# LangGraph
from langgraph.graph import StateGraph, START, END

# Вспомогательное
from IPython.display import display, Markdown

warnings.filterwarnings("ignore")

# Примечание: MultiQueryRetriever будет импортирован позже, когда понадобится

# Настройка API-ключей и модели
os.environ["OPENAI_API_KEY"] = "C1o2YnLwK1zzXdyoTLS33FnnE2ZoHpCO"  # <- Замените на ваш ключ
os.environ.setdefault("OPENAI_BASE_URL", "https://api.mistral.ai/v1")  # Mistral (OpenAI-совместимый)

# Выбор модели Mistral по умолчанию (можно заменить на нужную)
MODEL_NAME = os.getenv("MODEL_NAME", "mistral-large-latest")

print("OPENAI_BASE_URL:", os.getenv("OPENAI_BASE_URL"))
print("Модель:", MODEL_NAME)
print("Ключ установлен:", "OPENAI_API_KEY" in os.environ and os.environ["OPENAI_API_KEY"].startswith("sk-"))
api_key = os.environ.get("OPENAI_API_KEY")
print("Ключ установлен:", bool(api_key) and api_key != "MISTRAL_API_KEY")


# Часть 1: подготовка LLM, инструмента и состояния графа

# LLM (Mistral через OpenAI-совместимый интерфейс)
llm_news = ChatOpenAI(model=MODEL_NAME, temperature=0.3, timeout=120)

# Инструмент веб-поиска
search_tool = DuckDuckGoSearchRun()

# Состояние отчёта
class ReportState(TypedDict, total=False):
    topic: str
    research_results: List[str]
    draft: str
    critique: str
    final_report: str

# Вспомогательная функция: аккуратно форматируем список результатов
def _concat(items: List[str], sep: str = "\n\n") -> str:
    return sep.join([s for s in items if s and isinstance(s, str)])

# Узлы графа: researcher, writer, critic, researcher_extra, synthesizer

# Researcher: первичный поиск
researcher_prompt = ChatPromptTemplate.from_messages([
    ("system", "Ты аналитик ИИ-новостей. Извлеки краткие, полезные факты из данных поиска."),
    ("human", "Тема: {topic}\n\nСырые результаты:\n{raw_results}\n\nСформируй 3-6 маркеров с фактами (кратко).")
])

def researcher_node(state: ReportState) -> ReportState:
    query = f"{state['topic']} новости ИИ последние 7 дней"
    raw = search_tool.run(query)
    summary = (researcher_prompt | llm_news | StrOutputParser()).invoke({
        "topic": state["topic"],
        "raw_results": raw,
    })
    return {"research_results": [summary]}

# Writer: черновик отчёта
writer_prompt = ChatPromptTemplate.from_messages([
    ("system", "Ты технический писатель. Напиши связный абзац по пунктам."),
    ("human", "Тема: {topic}\n\nПункты:\n{bullets}\n\nТребования: 5-8 предложений, без воды, на русском.")
])

def writer_node(state: ReportState) -> ReportState:
    bullets = _concat(state.get("research_results", []))
    draft = (writer_prompt | llm_news | StrOutputParser()).invoke({
        "topic": state["topic"],
        "bullets": bullets,
    })
    return {"draft": draft}

# Critic: конструктивная критика
critic_prompt = ChatPromptTemplate.from_messages([
    ("system", "Ты строгий редактор. Дай конструктивную критику и улучшения."),
    ("human", "Черновик:\n{draft}\n\nДай 3-5 улучшений (структура, факты, ясность).")
])

def critic_node(state: ReportState) -> ReportState:
    critique = (critic_prompt | llm_news | StrOutputParser()).invoke({
        "draft": state.get("draft", ""),
    })
    return {"critique": critique}

# Researcher (доп. поиск): при необходимости расширяем фактуру
extra_prompt = ChatPromptTemplate.from_messages([
    ("system", "Определи, каких фактов не хватает, и дополни 2-4 маркерами."),
    ("human", "Тема: {topic}\nЧерновик:\n{draft}\n\nВерни только новые пункты (если нужны).")
])

def researcher_extra_node(state: ReportState) -> ReportState:
    query = f"{state['topic']} AI news site:arxiv.org OR site:blog.openai.com OR site:deepmind.google"
    raw = search_tool.run(query)
    extra = (extra_prompt | llm_news | StrOutputParser()).invoke({
        "topic": state["topic"],
        "draft": state.get("draft", ""),
    })
    combined = state.get("research_results", []) + [extra, raw[:800]]
    return {"research_results": combined}

# Synthesizer: финальный отчёт с учётом критики и доп. фактов
synth_prompt = ChatPromptTemplate.from_messages([
    ("system", "Ты финализируешь аналитический отчёт."),
    ("human", "Тема: {topic}\n\nЧерновик:\n{draft}\n\nКритика:\n{critique}\n\nДоп. факты:\n{facts}\n\nСобери финальный отчёт (7-10 предложений), исправь замечания, сохрани фактуру.")
])

def synthesizer_node(state: ReportState) -> ReportState:
    facts = _concat(state.get("research_results", []))
    final_report = (synth_prompt | llm_news | StrOutputParser()).invoke({
        "topic": state["topic"],
        "draft": state.get("draft", ""),
        "critique": state.get("critique", ""),
        "facts": facts,
    })
    return {"final_report": final_report}

# Построение графа с параллельными ветвями после writer
builder = StateGraph(ReportState)

builder.add_node("researcher", researcher_node)
builder.add_node("writer", writer_node)
builder.add_node("critic", critic_node)
builder.add_node("researcher_extra", researcher_extra_node)
builder.add_node("synthesizer", synthesizer_node)

# Линейная часть: START -> researcher -> writer
builder.add_edge(START, "researcher")
builder.add_edge("researcher", "writer")

# Fan-out: после writer запускаем параллельно critic и researcher_extra
builder.add_edge("writer", "critic")
builder.add_edge("writer", "researcher_extra")

# Fan-in: объединяем результаты в synthesizer
builder.add_edge("critic", "synthesizer")
builder.add_edge("researcher_extra", "synthesizer")

# Завершение
builder.add_edge("synthesizer", END)

graph_news = builder.compile()

# Визуализация графа (Mermaid)
try:
    mermaid = graph_news.get_graph().draw_mermaid()
    display(Markdown(f"""```mermaid\n{mermaid}\n```"""))
except Exception as e:
    print("Mermaid визуализация недоступна:", e)
    print("Узлы:", list(graph_news.get_graph().nodes()))
    print("Рёбра:", list(graph_news.get_graph().edges()))


# Запуск графа: финальный отчёт и замечания критика
inputs = {"topic": "последние новости в области искусственного интеллекта"}

start = time.time()
output = graph_news.invoke(inputs)
elapsed = time.time() - start

print("Время выполнения (с):", round(elapsed, 2))
print("\n--- Финальный отчёт ---\n")
print(output.get("final_report", "<пусто>"))
print("\n--- Замечания критика ---\n")
print(output.get("critique", "<пусто>"))
