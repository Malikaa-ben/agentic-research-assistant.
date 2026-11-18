# Agentic AI Research Assistant

Assistant de recherche **100 % local** : un agent **LangGraph** propulsé par **Llama 3.2 (Ollama)** qui choisit lui-même le bon outil parmi la recherche dans des **PDF (FAISS)**, une **calculatrice** et la **date**. Interface web avec **Streamlit**.

Aucune donnée n'est envoyée vers un service cloud : le modèle de langage, les embeddings et l'index vectoriel tournent sur ta machine.

## Fonctionnalités

- **Agent ReAct** (LangGraph) qui décide seul quel outil utiliser et peut en enchaîner plusieurs
- **RAG sur PDF** : découpage en passages, embeddings `nomic-embed-text`, recherche par similarité avec FAISS
- **Citation des sources** : fichier et page pour chaque passage utilisé
- **Calculatrice sécurisée** : évaluation des expressions via `ast`, sans jamais appeler `eval()`
- **Outil date et heure**
- **Mémoire de conversation** pendant la session
- **Transparence** : les outils appelés (arguments et résultats) sont affichés sous chaque réponse
- **Réponses dans la langue de l'utilisateur**

## Architecture

```
Question ─► Agent ReAct (LangGraph + Llama 3.2)
              ├─► search_documents ─► FAISS ◄─ embeddings nomic-embed-text ◄─ PDF
              ├─► calculator       (évaluation sécurisée via ast)
              └─► get_current_date
```

## Technologies

| Domaine | Outils |
|---|---|
| Agent | LangGraph (`create_react_agent`), LangChain |
| LLM | Llama 3.2 via Ollama (température 0) |
| Embeddings | `nomic-embed-text` via Ollama |
| Recherche vectorielle | FAISS (`faiss-cpu`) |
| Lecture des PDF | pypdf |
| Interface | Streamlit |
| Tests | pytest |

## Prérequis

- Python 3.10+
- [Ollama](https://ollama.com) installé et lancé

## Installation

```bash
# 1. Modèles Ollama
ollama pull llama3.2
ollama pull nomic-embed-text

# 2. Projet
git clone https://github.com/Malikaa-ben/agentic-research-assistant.git
cd agentic-research-assistant
python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # Linux / macOS
pip install -r requirements.txt
```

## Utilisation

```bash
streamlit run app.py
```

1. Ajoute tes PDF dans la barre latérale (ou directement dans `data/pdfs/`).
2. Clique sur **Indexer les PDF**.
3. Pose tes questions : les outils appelés sont affichés sous chaque réponse.

## Exemples

| Question | Outils utilisés |
|---|---|
| « Combien font 1847 * 23 + 560 ? » | `calculator` |
| « Quelle est la date aujourd'hui ? » | `get_current_date` |
| « Résume le chapitre 2 de mon PDF » | `search_documents` |
| « Quel est le budget dans le PDF ? Ajoute 20 % » | `search_documents` + `calculator` |

## Configuration

Les paramètres se trouvent dans `src/config.py` :

| Paramètre | Valeur par défaut | Rôle |
|---|---|---|
| `LLM_MODEL` | `llama3.2` | Modèle de langage Ollama |
| `EMBED_MODEL` | `nomic-embed-text` | Modèle d'embeddings |
| `TEMPERATURE` | `0.0` | Réponses déterministes |
| `CHUNK_SIZE` / `CHUNK_OVERLAP` | `800` / `120` | Taille des passages et chevauchement |
| `TOP_K` | `4` | Nombre de passages retournés par recherche |

## Tests

```bash
pytest                      # tests unitaires (sans Ollama)
python -m tests.scenarios   # scénarios multi-outils (avec Ollama)
```

## Structure du projet

```
app.py            Interface Streamlit
src/agent.py      Agent LangGraph
src/tools.py      Outils (documents, calcul, date)
src/rag.py        Indexation et recherche FAISS
src/config.py     Paramètres (modèles, chunks)
tests/            Tests unitaires et scénarios
data/pdfs/        Dossier des PDF à indexer
```

## Pistes d'amélioration

Mémoire persistante, recherche web, réponses en streaming, évaluation automatique des réponses.

## Licence

Projet distribué sous licence MIT (voir le fichier `LICENSE`).
