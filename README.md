# LangChain & LangGraph Notebooks

A hands-on, notebook-first path from **"call an LLM once"** to **multi-agent LangGraph workflows**, built on [LangChain](https://python.langchain.com/), [LangGraph](https://langchain-ai.github.io/langgraph/) and [Groq](https://groq.com/)-hosted open models.

Each notebook is written as a connected lesson rather than a code dump. Every one explains the idea first (what it does, why it exists, and what the alternatives are), then shows the code, then points out the common mistakes. Many include **⚠️ Bug watch** callouts that flag real rough edges instead of silently fixing them, plus exercises and a concept checklist at the end.

> **Good for:** learning LangChain/LangGraph properly, quick revision before interviews, and as a reference for the standard agentic workflow patterns.

---

## Contents

| # | Notebook | What you learn |
|---|----------|----------------|
| 01 | [`01_langchain_groq_concept_revision_notebook.ipynb`](01_langchain_groq_concept_revision_notebook.ipynb) | `ChatGroq` setup, `.invoke()` / `.batch()` / `.stream()` / `.astream_events()`, `temperature` & `max_tokens`, `PromptTemplate` vs `ChatPromptTemplate`, raw messages, caching, token-usage tracking, structured output with Pydantic |
| 02 | [`02_langchain_prompt_templates_revision.ipynb`](02_langchain_prompt_templates_revision.ipynb) | Prompt templates in depth: `PromptTemplate`, `ChatPromptTemplate`, template properties, `invoke` / `format` / `batch` / `save`, message types, and building a full template → model → response chain |
| 03 | [`03_langchain_runnables_extended.ipynb`](03_langchain_runnables_extended.ipynb) | `with_structured_output()`, tool calling with `@tool` and `bind_tools`, custom `Runnable` subclasses (caching, token guard), `RunnablePassthrough.assign()`, and a manual ReAct agent loop without `AgentExecutor` |
| 04 | [`04_langchain_lcel_revision.ipynb`](04_langchain_lcel_revision.ipynb) | LCEL and the Runnable contract: `RunnableLambda`, sequencing with `\|`, `invoke` / `batch` / `stream`, `RunnableParallel`, and combining runnables into full LLM chains |
| 05 | [`05_finance_agent_tutorial.ipynb`](05_finance_agent_tutorial.ipynb) | Rebuilding a tiered, tool-using finance agent: `init_chat_model`, `ToolRuntime` context injection, middleware (`wrap_model_call`, `dynamic_prompt`, `wrap_tool_call`), `ToolStrategy` structured output, checkpointer memory, `create_agent` |
| 06 | [`06_langgraph_fundamentals_tutorial.ipynb`](06_langgraph_fundamentals_tutorial.ipynb) | LangGraph core: state, nodes, edges, reducers, three ways to define state (`dict` / `TypedDict` / Pydantic), `MessagesState`, runtime context, conditional edges, `Command`, and `Send` fan-out |
| 07 | [`07_augmented_llm_lesson.ipynb`](07_augmented_llm_lesson.ipynb) | The **augmented LLM**: tools, structured output, and prompt chaining orchestrated with LangGraph |
| 08 | [`08_langgraph_routing_and_parallelization.ipynb`](08_langgraph_routing_and_parallelization.ipynb) | Two workflow patterns: **routing** (classify, then take one branch) and **parallelization** (fan-out, then an implicit fan-in join) |
| 09 | [`09_langgraph_orchestrator_worker_lesson.ipynb`](09_langgraph_orchestrator_worker_lesson.ipynb) | **Orchestrator–worker** multi-agent research pipeline: planner, parallel workers via `Send`, synthesizer, reducers, rate limiting, and `tenacity` retries |
| 10 | [`10_evaluator_optimizer_lesson.ipynb`](10_evaluator_optimizer_lesson.ipynb) | **Evaluator–optimizer** loop: generate → critique → refine → repeat, with Pydantic-typed feedback and a conditional-edge stop condition |
| — | [`NOVA_interview_deep_dive.ipynb`](NOVA_interview_deep_dive.ipynb) | Capstone walkthrough of **NOVA**, a Streamlit + LangGraph research assistant: intent and search graphs, human-in-the-loop `interrupt()`, SPECTER reranking and clustering, and a single-PDF RAG chatbot |

### Suggested learning path

```
01 → 02 → 03 → 04        LangChain fundamentals: models, prompts, runnables, LCEL
        ↓
   05                    Agents: tools, middleware, memory, structured output
        ↓
06 → 07                  LangGraph basics and the augmented LLM
        ↓
08 → 09 → 10             Workflow patterns: routing, parallelization,
                         orchestrator–worker, evaluator–optimizer
        ↓
   NOVA                  Putting it all together in a real app
```

The workflow patterns in 07–10 follow the vocabulary from Anthropic's [*Building Effective Agents*](https://www.anthropic.com/engineering/building-effective-agents).

---

## Getting started

### 1. Clone and create an environment

```bash
git clone https://github.com/sagarbisht123/<repo-name>.git
cd <repo-name>

python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
```

### 2. Install dependencies

Notebooks 01–10 need:

```bash
pip install jupyter langchain langchain-core langchain-groq langchain-community \
            langgraph python-dotenv pydantic tenacity tiktoken grandalf
```

- `grandalf` is optional and only needed for `draw_ascii()` graph rendering in notebook 06.
- `fastapi` is imported in notebook 03 for one example; install it if you plan to run that cell.
- The NOVA notebook uses a much larger stack (`streamlit`, `sentence-transformers`, `chromadb`/`langchain-chroma`, `langchain-tavily`, `pymupdf`, and others). Its code cells come from the NOVA repo and are meant to be run inside NOVA's own environment, so treat that notebook as a guided read unless you set up the project.

### 3. Add your Groq API key

Get a free key from the [Groq console](https://console.groq.com/keys), then create a `.env` file in the project root:

```bash
GROQ_API_KEY=your_key_here
```

`.env` should be in your `.gitignore`. Never commit it.

### 4. Run

```bash
jupyter notebook
```

Open any notebook and run the cells top to bottom.

---

## Models used

| Model | Where |
|-------|-------|
| `llama-3.3-70b-versatile` | Notebooks 01–05, NOVA |
| `llama-3.1-8b-instant` | Notebook 05 (cheaper tier in the tiered agent) |
| `openai/gpt-oss-120b` | Notebooks 06–08, 10, NOVA |
| `openai/gpt-oss-20b` | Notebooks 09–10 |

Groq retires models from time to time. If a call fails with a model-not-found error, check the [current model list](https://console.groq.com/docs/models) and update the model string. Because everything goes through LangChain's common interface, swapping providers (for example `ChatOpenAI` or `ChatAnthropic`) is mostly a one-line change.

---

## Repository structure

```
.
├── 01_langchain_groq_concept_revision_notebook.ipynb
├── 02_langchain_prompt_templates_revision.ipynb
├── 03_langchain_runnables_extended.ipynb
├── 04_langchain_lcel_revision.ipynb
├── 05_finance_agent_tutorial.ipynb
├── 06_langgraph_fundamentals_tutorial.ipynb
├── 07_augmented_llm_lesson.ipynb
├── 08_langgraph_routing_and_parallelization.ipynb
├── 09_langgraph_orchestrator_worker_lesson.ipynb
├── 10_evaluator_optimizer_lesson.ipynb
├── NOVA_interview_deep_dive.ipynb
└── README.md
```

---

## Notes

- Notebooks call a live LLM API, so outputs will vary between runs. Use `temperature=0` where you want repeatable results.
- A few notebooks were drafted without network access to Groq, so some code cells may be saved without outputs. Run them with your own key to see real results.
- Bug-watch callouts are intentional. Try spotting the issue yourself before reading the explanation.

## Contributing

Issues and PRs are welcome, especially for fixes to outdated APIs or model names, and for new exercises.

## Author

**Sagar** ([@sagarbisht123](https://github.com/sagarbisht123)), BS-MS Mathematics at IISER Berhampur, working at the intersection of mathematics and machine learning.
