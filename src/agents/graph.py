from typing import Literal
from langgraph.graph import StateGraph, END
from src.agents.nodes import AgentState, planner_node, rag_retriever_node, tool_executor_node, synthesizer_node
from config.settings import Config

# Enhanced Synthesizer: Use OpenAI if Key exists
HAS_OPENAI = bool(Config.OPENAI_API_KEY)

def enhanced_synthesizer_node(state: AgentState) -> AgentState:
    if HAS_OPENAI:
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
    else:
        # Fallback to standard local logic
        return synthesizer_node(state)
    
    return state

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
