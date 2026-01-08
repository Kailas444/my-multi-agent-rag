from typing import Literal
from langgraph.graph import StateGraph, END
from src.agents.nodes import AgentState, planner_node, rag_retriever_node, tool_executor_node, synthesizer_node
from config.settings import Config

# Check keys safely
HAS_OPENAI = bool(Config.OPENAI_API_KEY)
HAS_GOOGLE = bool(Config.GOOGLE_API_KEY)

def enhanced_synthesizer_node(state: AgentState) -> AgentState:
    # Priority: Try Google -> Try OpenAI -> Fallback to Local
    if HAS_GOOGLE:
        try:
            from langchain_google_genai import ChatGoogleGenerativeAI
            llm = ChatGoogleGenerativeAI(api_key=Config.GOOGLE_API_KEY, model="gemini-pro")
            
            prompt = f"""
            Reasoning: {state['reasoning_trace']}
            Context: {state.get('retrieved_context', 'None')}
            Tool Output: {state.get('tool_output', 'None')}
            Question: {state['query']}
            Answer concisely.
            """
            state["final_answer"] = llm.invoke(prompt).content
            return state
        except Exception as e:
            # If Google fails, print error in trace and fallback
            state["reasoning_trace"].append(f"Google API Error: {e}. Trying Fallback.")
            
    elif HAS_OPENAI:
        try:
            from langchain_openai import ChatOpenAI
            llm = ChatOpenAI(api_key=Config.OPENAI_API_KEY, temperature=0)
            prompt = f"""
            Reasoning: {state['reasoning_trace']}
            Context: {state.get('retrieved_context', 'None')}
            Tool Output: {state.get('tool_output', 'None')}
            Question: {state['query']}
            Answer concisely.
            """
            state["final_answer"] = llm.invoke(prompt).content
            return state
        except Exception as e:
            state["reasoning_trace"].append(f"OpenAI API Error: {e}. Trying Fallback.")

    # Final Fallback (Local Logic)
    return synthesizer_node(state)

def decide_route(state: AgentState) -> str:
    return "tool_executor" if state["plan"] == "use_tool" else "rag_retriever"

def rag_decision(state: AgentState) -> str:
    return "tool_executor" if state["plan"] == "use_tool" else "synthesizer"

workflow = StateGraph(AgentState)
workflow.add_node("planner", planner_node)
workflow.add_node("rag_retriever", rag_retriever_node)
workflow.add_node("tool_executor", tool_executor_node)
workflow.add_node("synthesizer", enhanced_synthesizer_node)

workflow.set_entry_point("planner")
workflow.add_conditional_edges("planner", decide_route, {"rag_retriever": "rag_retriever", "tool_executor": "tool_executor"})
workflow.add_conditional_edges("rag_retriever", rag_decision, {"tool_executor": "tool_executor", "synthesizer": "synthesizer"})
workflow.add_edge("tool_executor", "synthesizer")
workflow.add_edge("synthesizer", END)

app = workflow.compile()
