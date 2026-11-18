"""Scénarios manuels (nécessite Ollama) : python -m tests.scenarios"""
from src.agent import build_agent, run_agent

SCENARIOS = [
    ("Calcul seul", "Combien font 1847 * 23 + 560 ?", {"calculator"}),
    ("Date seule", "Quelle est la date aujourd'hui ?", {"get_current_date"}),
    ("Documents seuls", "Résume les points clés de mes documents.", {"search_documents"}),
    ("Multi-outils", "Selon mes documents, quel est le budget total ? Multiplie-le par 1.2.",
     {"search_documents", "calculator"}),
]

if __name__ == "__main__":
    agent = build_agent()
    for name, question, expected in SCENARIOS:
        out = run_agent(agent, question)
        used = {s["tool"] for s in out["steps"]}
        status = "OK " if expected <= used else "KO "
        print(f"{status}| {name} | outils utilisés : {sorted(used)}")
        print(f"    -> {out['answer'][:200]}\n")
