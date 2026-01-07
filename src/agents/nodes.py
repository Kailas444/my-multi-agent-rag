import re
from typing import TypedDict, List, Dict, Any
from src.rag.rag_engine import RAGSystem
from src.tools.custom_tools import weather_tool_func, calculator_tool_func
from langchain_community.tools import DuckDuckGoSearchRun

search_tool = DuckDuckGoSearchRun()
rag_system = RAGSystem()

class AgentState(TypedDict):
    query: str
    plan: str
    reasoning_trace: List[str]
    retrieved_context: str
    citations: List[Dict]
    tool_output: str
    final_answer: str

def planner_node(state: AgentState) -> AgentState:
    query = state["query"].lower()
    plan = "UNKNOWN"
    thought = f"Analyzing: '{state['query']}'. "
    
    if "weather" in query or "temperature" in query:
        plan = "use_tool"
        state["reasoning_trace"].append(thought + "Intent: Weather. Action: Call Weather.")
    elif "calculate" in query or re.search(r"\d+[\+\-\*\/]\d+", query):
        plan = "use_tool"
        state["reasoning_trace"].append(thought + "Intent: Math. Action: Call Calculator.")
    else:
        plan = "retrieve_rag"
        state["reasoning_trace"].append(thought + "Intent: Domain Query. Action: RAG.")
        
    state["plan"] = plan
    return state

def rag_retriever_node(state: AgentState) -> AgentState:
    state["reasoning_trace"].append("Executing RAG...")
    context, citations = rag_system.retrieve(state["query"])
    
    if not context:
        state["reasoning_trace"].append("RAG failed. Fallback to Search.")
        state["plan"] = "use_tool"
    else:
        state["retrieved_context"] = context
        state["citations"] = citations
        state["reasoning_trace"].append(f"RAG Success: Retrieved {len(citations)} chunks.")
    return state

def tool_executor_node(state: AgentState) -> AgentState:
    q = state["query"]
    result = ""
    
    if "weather" in q.lower():
        q_clean = q.rstrip("?,!. ")
        loc = "Unknown"
        if " in " in q_clean: loc = q_clean.split(" in ")[1]
        elif " at " in q_clean: loc = q_clean.split(" at ")[1]
        elif "weather" in q_clean: loc = q_clean.split("weather")[1]
        loc = loc.strip()
        result = weather_tool_func(loc)
        state["reasoning_trace"].append(f"Tool Used: Weather(loc={loc})")
        
    elif "calculate" in q.lower():
        expr = re.sub(r"[^\d+\-*/().]", "", q)
        result = calculator_tool_func(expr)
        state["reasoning_trace"].append(f"Tool Used: Calculator(expr={expr})")
    else:
        result = search_tool.run(q)
        state["reasoning_trace"].append("Tool Used: DuckDuckGo Search")
        
    state["tool_output"] = result
    return state

def synthesizer_node(state: AgentState) -> AgentState:
    state["reasoning_trace"].append("Synthesizing final answer...")
    
    if state.get("retrieved_context"):
        state["final_answer"] = f"Based on documents: {state['retrieved_context'][:300]}..."
    elif state.get("tool_output"):
        state["final_answer"] = f"Based on tools: {state['tool_output']}"
    else:
        state["final_answer"] = "I couldn't find an answer."
    return state
