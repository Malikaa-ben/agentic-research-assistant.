"""Agent ReAct LangGraph + Llama 3.2 (Ollama)."""
from langchain_core.messages import AIMessage, ToolMessage
from langchain_ollama import ChatOllama
from langgraph.prebuilt import create_react_agent

from .config import LLM_MODEL, TEMPERATURE
from .tools import TOOLS

SYSTEM_PROMPT = """Tu es un assistant de recherche. Réponds dans la langue de l'utilisateur.
Choisis toi-même le bon outil :
- search_documents : pour toute question sur le contenu des PDF ;
- calculator : pour tout calcul (n'effectue jamais un calcul de tête) ;
- get_current_date : pour la date ou l'heure.
Tu peux enchaîner plusieurs outils. Cite les sources (fichier, page) quand tu utilises les documents.
Si l'information est introuvable dans les documents, dis-le clairement."""


def build_agent():
    llm = ChatOllama(model=LLM_MODEL, temperature=TEMPERATURE)
    return create_react_agent(llm, TOOLS, prompt=SYSTEM_PROMPT)


def run_agent(agent, question: str, history: list | None = None) -> dict:
    """Exécute l'agent. Retourne {'answer': str, 'steps': [{'tool','args','output'}]}."""
    messages = (history or []) + [("user", question)]
    result = agent.invoke({"messages": messages})

    steps, pending = [], {}
    for m in result["messages"]:
        if isinstance(m, AIMessage):
            for call in m.tool_calls:
                pending[call["id"]] = {"tool": call["name"], "args": call["args"], "output": ""}
        elif isinstance(m, ToolMessage) and m.tool_call_id in pending:
            pending[m.tool_call_id]["output"] = m.content
            steps.append(pending[m.tool_call_id])

    return {"answer": result["messages"][-1].content, "steps": steps}
