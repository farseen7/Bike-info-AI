import streamlit as st
import os
import re
import json
import uuid
import duckdb
import chromadb
from typing import TypedDict, Optional, List, Annotated
from pydantic import BaseModel, Field, AliasChoices
from typing_extensions import Literal
from groq import Groq
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from sentence_transformers import SentenceTransformer

# --- CONFIG ---
MAX_HISTORY_MESSAGES = 20    # total turns (user+assistant) kept in memory per thread
MAX_HISTORY_FOR_PROMPT = 6   # how many of those are actually fed back into prompts

# --- 1. STATE & SCHEMA DEFINITIONS ---


def _keep_recent_history(existing: Optional[List[dict]], new: Optional[List[dict]]) -> List[dict]:
    """Reducer for chat_history: append new turns, then cap the total length so
    conversation memory doesn't grow (and inflate prompt size) forever."""
    combined = (existing or []) + (new or [])
    return combined[-MAX_HISTORY_MESSAGES:]


class AgentState(TypedDict):
    user_query: str
    chat_history: Annotated[List[dict], _keep_recent_history]
    route: str
    target_entity: List[str]
    sql_query: Optional[str]
    sql_data: Optional[List[dict]]
    vector_data: Optional[List[str]]
    sql_error: Optional[str]
    retry_count: int
    final_response: Optional[str]


class RouteResponse(BaseModel):
    next_step: Literal["SQL", "VECTOR", "HYBRID"] = Field(
        validation_alias=AliasChoices("next_step", "route"),
        description="The routing path based on query intent."
    )
    target_entity: List[str] = Field(
        default_factory=list,
        description="Extracted bike model or company names from query."
    )
    reasoning: str = Field(
        default="No reasoning provided.",
        description="Brief explanation of why this route was selected."
    )


# --- 2. CACHED RESOURCES & CLIENT SETUP ---

@st.cache_resource
def get_groq_client():
    api_key = st.secrets.get("GROQ_API_KEY", os.environ.get("GROQ_API_KEY"))
    if not api_key:
        return None
    return Groq(api_key=api_key)


@st.cache_resource
def get_duckdb_connection():
    token = st.secrets.get("MOTHERDUCK_TOKEN", os.environ.get("motherduck_token"))
    if not token:
        st.warning("MotherDuck token not found. Falling back to in-memory DuckDB.")
        return duckdb.connect()
    try:
        con = duckdb.connect('md:bikes', config={'motherduck_token': token})
        return con
    except Exception as e:
        st.error(f"Failed to connect to MotherDuck: {e}. Using in-memory DuckDB.")
        return duckdb.connect()


@st.cache_resource
def load_chromadb_collections():
    chroma_client = chromadb.PersistentClient(path="./chroma_db")
    collection_feature = chroma_client.get_collection(name="bike_features")
    collection_reviews = chroma_client.get_collection(name="bike_reviews")
    return collection_feature, collection_reviews


@st.cache_resource
def get_embedder():
    return SentenceTransformer("all-MiniLM-L6-v2")


@st.cache_resource
def get_checkpointer():
    # In-memory checkpointer: gives each chat thread persistent state (including
    # chat_history) for the life of the process. Swap for SqliteSaver/PostgresSaver
    # if you need memory to survive an app restart / multiple server instances.
    return MemorySaver()


# Initialize global connections
client = get_groq_client()
con = get_duckdb_connection()
collection_feature, collection_reviews = load_chromadb_collections()
embedder = get_embedder()


# --- 3. HELPER FUNCTIONS ---

def _format_history_for_prompt(chat_history: List[dict], limit: int = MAX_HISTORY_FOR_PROMPT) -> str:
    """Render the last few turns as plain text so an LLM prompt can resolve
    references like 'it', 'that bike', 'the one you mentioned', etc."""
    if not chat_history:
        return "No prior conversation."
    recent = chat_history[-limit:]
    lines = []
    for turn in recent:
        role = "User" if turn.get("role") == "user" else "Assistant"
        lines.append(f"{role}: {turn.get('content', '')}")
    return "\n".join(lines)


_SQL_BLOCKLIST = re.compile(
    r"\b(drop|delete|update|insert|alter|attach|detach|copy|pragma|create|grant|call|truncate|export|import)\b",
    re.IGNORECASE,
)


def _is_safe_select(sql: str) -> bool:
    """Only allow a single, read-only SELECT (optionally with a leading CTE).
    The SQL text here is LLM output, so it must never be trusted blindly."""
    if not sql:
        return False
    stripped = sql.strip().rstrip(";").strip()
    if not re.match(r"^\s*(with\b.*?)?select\b", stripped, re.IGNORECASE | re.DOTALL):
        return False
    if ";" in stripped:  # no stacked statements
        return False
    if _SQL_BLOCKLIST.search(stripped):
        return False
    return True


def route_decision(state: AgentState):
    route = state["route"]
    if route == "SQL":
        return "sql_developer"
    elif route == "VECTOR":
        return "vector_search"
    elif route == "HYBRID":
        return ["sql_developer", "vector_search"]
    return "vector_search"


def check_sql_status(state: AgentState):
    sql_error = state.get("sql_error")
    retry_count = state.get("retry_count", 0)
    if sql_error and retry_count < 3:
        return "sql_developer"
    return "synthesizer"


# --- 4. NODE DEFINITIONS ---

def supervisor_node(state: AgentState) -> AgentState:
    user_query = state["user_query"]
    history_entry = [{"role": "user", "content": user_query}]

    # Reset per-turn fields so leftovers from a *previous* turn (e.g. sql_data
    # from an earlier HYBRID query) can't leak into a turn that never touches
    # that branch this time around.
    reset_fields = {
        "sql_query": None,
        "sql_data": None,
        "vector_data": None,
        "sql_error": None,
        "retry_count": 0,
        "chat_history": history_entry,
    }

    if client is None:
        return {**reset_fields, "route": "VECTOR", "target_entity": []}

    history_text = _format_history_for_prompt(state.get("chat_history", []))

    prompt = f"""
You are a routing supervisor for a bike information assistant. Analyze the user's
latest query IN THE CONTEXT of the recent conversation (resolve pronouns like
"it" / "that one" using the conversation history) and return ONLY a JSON object
matching this schema:

{{
  "route": "SQL" | "VECTOR" | "HYBRID",
  "target_entity": ["bike model or brand names"],
  "reasoning": "explanation for routing choice"
}}

ROUTING RULES:
1. Choose "SQL" for structured specs, prices, mileage, features.
2. Choose "VECTOR" for user reviews, opinions, pros/cons.
3. Choose "HYBRID" for queries combining both specs and reviews.

ENTITY EXTRACTION & NORMALIZATION:
- Correct common bike name misnomers/typos to standard names if obvious (e.g., "CBR 350" or "CBR350" -> "CB350" / "Honda CB350").
- If unsure, include both the original query term and the closest match in target_entity.
- If the latest query refers back to a bike discussed earlier (e.g. "what about its mileage?"), reuse that bike as the target_entity.

RECENT CONVERSATION:
{history_text}

LATEST USER QUERY: "{user_query}"
"""

    try:
        chat_completion = client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model="openai/gpt-oss-20b",
            response_format={"type": "json_object"},
            temperature=0.0,
        )
        decision = RouteResponse.model_validate_json(chat_completion.choices[0].message.content)
        return {
            **reset_fields,
            "route": decision.next_step,
            "target_entity": decision.target_entity,
        }
    except Exception:
        # If routing fails for any reason, degrade gracefully to a vector
        # search rather than crashing the whole graph.
        return {**reset_fields, "route": "VECTOR", "target_entity": []}


def sql_developer_node(state: AgentState) -> AgentState:
    if con is None:
        return {
            "sql_query": None,
            "sql_data": [],
            "sql_error": "No DuckDB connection",
            "retry_count": state.get("retry_count", 0) + 1,
        }

    user_query = state["user_query"]
    target_entities = state.get("target_entity", [])
    sql_error = state.get("sql_error")
    retry_count = state.get("retry_count", 0)
    previous_sql = state.get("sql_query", "")

    schema_info = """
    Table 1: bike_features
    Columns: "Variant Name", "Company Name", "On-road prize", "Engine Type", "Displacement", "Max Torque", "No. of Cylinders", "Cooling System", "City Mileage", "Highway Mileage", "Body Type", "0-100 Kmph (ec)", "Peak Power", "Transmission"

    Table 2: bikes_reviews
    Columns: Varient_Name, Average_stars, Review_title, User_rating, Review_description
    """

    if sql_error:
        prompt = f"""Fix this DuckDB SQL error.
SCHEMA:
{schema_info}
FAILED QUERY: {previous_sql}
ERROR: {sql_error}
QUESTION: {user_query}
Return ONLY a single read-only SELECT statement, inside a ```sql ... ``` block. No DDL/DML."""
    else:
        prompt = f"""Write a single DuckDB SQL SELECT query.
SCHEMA:
{schema_info}

RULES:
1. Double quote column names with spaces (e.g. "Variant Name").
2. ALWAYS use loose wildcard matching for bike models using ILIKE with % between letters and numbers. Example: for 'CB350', write `WHERE "Variant Name" ILIKE '%CB%350%' OR "Variant Name" ILIKE '%Honda%'`.
3. Include `LIMIT 10`.
4. Read-only SELECT only. No DDL/DML, no multiple statements.
Return ONLY raw SQL inside ```sql ... ``` block.

QUESTION: "{user_query}"
TARGET ENTITIES: {target_entities}
"""

    try:
        response = client.chat.completions.create(
            model="openai/gpt-oss-20b",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
        )
        raw_text = response.choices[0].message.content
    except Exception as e:
        return {
            "sql_query": previous_sql,
            "sql_data": None,
            "sql_error": f"LLM call failed: {e}",
            "retry_count": retry_count + 1,
        }

    sql_match = re.search(r"```sql\s*(.*?)\s*```", raw_text, re.DOTALL)
    sql_query = sql_match.group(1).strip() if sql_match else raw_text.replace("```", "").strip()

    if not _is_safe_select(sql_query):
        return {
            "sql_query": sql_query,
            "sql_data": None,
            "sql_error": "Generated SQL failed the read-only SELECT safety check.",
            "retry_count": retry_count + 1,
        }

    try:
        query_results = con.execute(sql_query).df().to_dict(orient="records")

        # FALLBACK: if the query succeeded but returned 0 rows, run a broad
        # fuzzy keyword search using a parameterized query (never string-format
        # LLM-derived text straight into SQL).
        if not query_results and target_entities:
            clean_entity = re.sub(r"[^a-zA-Z0-9]", "", target_entities[0])
            if clean_entity:
                fallback_pattern = "%" + "%".join(list(clean_entity)) + "%"
                fallback_sql = 'SELECT * FROM bike_features WHERE "Variant Name" ILIKE ? LIMIT 5;'
                query_results = con.execute(fallback_sql, [fallback_pattern]).df().to_dict(orient="records")
                sql_query = fallback_sql

        return {"sql_query": sql_query, "sql_data": query_results, "sql_error": None, "retry_count": retry_count}
    except Exception as e:
        return {"sql_query": sql_query, "sql_data": None, "sql_error": str(e), "retry_count": retry_count + 1}


def vector_search_node(state: AgentState) -> AgentState:
    user_query = state["user_query"]
    target_entities = state.get("target_entity", [])

    if len(user_query.strip()) < 3:
        return {"vector_data": []}

    query_vector = embedder.encode([user_query]).tolist()
    formatted_docs = []

    # Attempt 1: query with a metadata filter scoped to the target entity/entities
    where_filter = {"Variant Name": {"$in": target_entities}} if target_entities else None

    try:
        review_results = collection_reviews.query(query_embeddings=query_vector, n_results=3, where=where_filter)
        rev_docs = review_results.get("documents", [[]])[0]
        for doc in rev_docs:
            formatted_docs.append(f"\U0001F4AC [REVIEW]: {doc[:250]}")
    except Exception:
        pass

    # Attempt 2: fall back to pure semantic search with no metadata filter
    if not formatted_docs:
        try:
            fallback_reviews = collection_reviews.query(query_embeddings=query_vector, n_results=3)
            rev_docs = fallback_reviews.get("documents", [[]])[0]
            for doc in rev_docs:
                formatted_docs.append(f"\U0001F4AC [REVIEW]: {doc[:250]}")
        except Exception:
            pass

    return {"vector_data": formatted_docs}


def synthesizer_node(state: AgentState) -> AgentState:
    user_query = state.get("user_query", "")
    route = state.get("route", "")
    sql_data = state.get("sql_data", None)
    vector_data = state.get("vector_data", None)
    chat_history = state.get("chat_history", [])

    if sql_data and isinstance(sql_data, list):
        sql_context_str = json.dumps(sql_data[:5], indent=2, default=str)
    else:
        sql_context_str = "No database records found."

    if vector_data and isinstance(vector_data, list):
        vector_context_str = "\n".join(vector_data[:3])[:1000]
    else:
        vector_context_str = "No relevant reviews found."

    history_text = _format_history_for_prompt(chat_history)

    if client is None:
        final_response = "Please enter your Groq API Key in the sidebar or Streamlit secrets."
        return {
            "final_response": final_response,
            "chat_history": [{"role": "assistant", "content": final_response}],
        }

    synthesizer_prompt = f"""
You are a specialized Bike Information AI Assistant. You CAN ONLY answer questions related to motorcycles, scooters, bike specifications, features, mileage, prices, and user reviews.

RECENT CONVERSATION (for continuity - refer back to it naturally if relevant):
{history_text}

Route Type: {route}
Latest User Query: {user_query}

SQL Data Context:
{sql_context_str}

Vector Review Context:
{vector_context_str}

STRICT RESPONSE RULES:
1. OFF-TOPIC & SMALL TALK REJECTION: If the user query is unrelated to bikes (e.g., general knowledge, coding, math, personal questions, schedules), respond strictly with:
   "I can only answer bike-related questions. Please ask me about bike specifications, features, prices, mileage, or user reviews!"

2. GREETINGS: If the user says a basic greeting like "hi" or "hello", greet them back briefly in one sentence and inform them that you can help with bike-related queries.

3. DATA-BASED ANSWERS: If the query is about bikes and relevant data exists in the context above, construct a clear, helpful answer using ONLY that data.

4. MISSING DATA: If the query is bike-related but no records match in the context, state clearly that you don't have records for that specific bike model in your database.

5. CONTINUITY: If the latest query refers back to something discussed earlier ("it", "that one", "the first bike"), use the conversation history to understand what's being asked, but still only state facts backed by the data contexts above.
"""

    try:
        response = client.chat.completions.create(
            model="openai/gpt-oss-20b",
            messages=[{"role": "user", "content": synthesizer_prompt}],
            temperature=0.0,
        )
        final_response = response.choices[0].message.content
    except Exception as e:
        final_response = f"Sorry, I hit an error generating a response: {e}"

    return {
        "final_response": final_response,
        "chat_history": [{"role": "assistant", "content": final_response}],
    }


# --- 5. LANGGRAPH WORKFLOW ---

workflow = StateGraph(AgentState)
workflow.add_node("supervisor", supervisor_node)
workflow.add_node("sql_developer", sql_developer_node)
workflow.add_node("vector_search", vector_search_node)
workflow.add_node("synthesizer", synthesizer_node)

workflow.add_edge(START, "supervisor")
workflow.add_conditional_edges("supervisor", route_decision, ["sql_developer", "vector_search"])
workflow.add_conditional_edges("sql_developer", check_sql_status, {"sql_developer": "sql_developer", "synthesizer": "synthesizer"})
workflow.add_edge("vector_search", "synthesizer")
workflow.add_edge("synthesizer", END)

app = workflow.compile(checkpointer=get_checkpointer())


# --- 6. STREAMLIT INTERFACE ---

st.set_page_config(page_title="Bike AI Assistant", page_icon="🏍️", layout="wide")
st.title("🏍️ Multi-Agent Bike Assistant")

if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())

if "messages" not in st.session_state:
    st.session_state.messages = []

with st.sidebar:
    st.header("⚙️ Configuration")
    key_input = st.text_input(
        "Groq API Key",
        type="password",
        value=st.secrets.get("GROQ_API_KEY", os.environ.get("GROQ_API_KEY", "")),
    )
    if key_input:
        os.environ["GROQ_API_KEY"] = key_input

    st.divider()
    st.caption(f"Conversation memory id: `{st.session_state.thread_id[:8]}`")
    if st.button("🗑️ Clear conversation memory"):
        st.session_state.thread_id = str(uuid.uuid4())
        st.session_state.messages = []
        st.rerun()

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if prompt := st.chat_input("Ask a question about bike specs or reviews..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.status("🤖 Executing Agents...", expanded=True) as status:
            if client is None and os.environ.get("GROQ_API_KEY"):
                get_groq_client.clear()
                client = get_groq_client()

            if client is None:
                st.error("GROQ_API_KEY missing.")
                status.update(label="Failed", state="error", expanded=False)
                res_text = "Please enter your Groq API Key in the sidebar or Streamlit secrets."
            else:
                config = {"configurable": {"thread_id": st.session_state.thread_id}}
                out = app.invoke({"user_query": prompt}, config=config)
                status.update(label="Complete", state="complete", expanded=False)
                res_text = out.get("final_response", "No response generated.")

        st.markdown(res_text)
        st.session_state.messages.append({"role": "assistant", "content": res_text})
