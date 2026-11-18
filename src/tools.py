"""Outils mis à disposition de l'agent."""
import ast
import operator
from datetime import datetime

from langchain_core.tools import tool

from .config import TOP_K
from .rag import get_store

_OPS = {
    ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
    ast.Div: operator.truediv, ast.Pow: operator.pow, ast.Mod: operator.mod,
    ast.FloorDiv: operator.floordiv, ast.USub: operator.neg, ast.UAdd: operator.pos,
}


def _eval(node):
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _OPS:
        return _OPS[type(node.op)](_eval(node.left), _eval(node.right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in _OPS:
        return _OPS[type(node.op)](_eval(node.operand))
    raise ValueError("Expression non autorisée")


def safe_calculate(expression: str):
    """Évalue une expression arithmétique sans utiliser eval()."""
    return _eval(ast.parse(expression.replace("^", "**"), mode="eval").body)


@tool
def calculator(expression: str) -> str:
    """Évalue une expression mathématique (+, -, *, /, **, %). Exemple : '(12.5 * 8) / 3'."""
    try:
        return str(safe_calculate(expression))
    except Exception as e:
        return f"Erreur de calcul : {e}"


@tool
def search_documents(query: str) -> str:
    """Cherche des passages pertinents dans les PDF indexés. À utiliser pour toute
    question portant sur le contenu des documents de l'utilisateur."""
    store = get_store()
    if store is None:
        return "Aucun document indexé. Ajoutez des PDF puis lancez l'indexation."
    results = store.similarity_search(query, k=TOP_K)
    return "\n\n".join(
        f"[{d.metadata.get('source', '?').split('/')[-1]} p.{d.metadata.get('page', 0) + 1}]\n{d.page_content}"
        for d in results
    )


@tool
def get_current_date() -> str:
    """Retourne la date et l'heure actuelles."""
    return datetime.now().strftime("%A %d %B %Y, %H:%M")


TOOLS = [search_documents, calculator, get_current_date]
