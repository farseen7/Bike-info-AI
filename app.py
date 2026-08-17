import streamlit as st
import os
import json
import duckdb
import chromadb
from typing import TypedDict, Optional, List
from pydantic import BaseModel, Field, AliasChoices
from typing_extensions import Literal
from groq import Groq
from langgraph.graph import StateGraph, START, END
from sentence_transformers import SentenceTransformer

class AgentState(TypedDict):
    user_query: str
    route: str
    target_entity: List[str]
    sql_query: Optional[str]
    sql_data: Optional[List[dict]]
    vector_docs: Optional[List[str]]
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
        con = duckdb.connect(f'md:bikes', config={'motherduck_token': token})
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


# Initialize global connections
client = get_groq_client()
con = get_duckdb_connection()
collection_feature, collection_reviews = load_chromadb_collections()
embedder = get_embedder()


# --- 3. HELPER ROUTING FUNCTIONS ---

def route_decision(state: AgentState):
    route = state["route"]
    if route == "SQL":
        return "sql_developer"
    elif route == "VECTOR":
        return "vector_search"
    elif route == "HYBRID":
        return ["sql_developer", "vector_search"]

def check_sql_status(state: AgentState):
    sql_error = state.get("sql_error")
    retry_count = state.get("retry_count", 0)
    if sql_error and retry_count < 3:
        st.write(f"⚠️ Retrying SQL Node (Attempt {retry_count + 1}/3)")
        return "sql_developer"
    return "synthesizer"


# --- 4. NODE DEFINITIONS ---

def supervisor_node(state: AgentState) -> AgentState:
    if client is None:
        return {"route": "VECTOR", "target_entity": [], "retry_count": 0, "sql_error": None}

    user_query = state["user_query"]

    prompt = f"""You are the Supervisor Agent for a Bike Database.

STEP 1: Identify and extract ALL bike brand names and model names from the query into `target_entity`. Return empty list `[]` if none.
STEP 2: Classify route:
- "SQL": Counts, price filters, mileage, exact numeric specs.
- "VECTOR": Reviews, subjective comfort, riding experience.
- "HYBRID": Both specs and qualitative reviews requested.

USER QUERY: "{user_query}"
Return JSON adhering strictly to RouteResponse schema.
"""

    chat_completion = client.chat.completions.create(
        messages=[{"role": "user", "content": prompt}],
        model="openai/gpt-oss-20b",
        response_format={"type": "json_object"},
        temperature=0.0,
    )

    decision = RouteResponse.model_validate_json(chat_completion.choices[0].message.content)

    return {
        "route": decision.next_step,
        "target_entity": decision.target_entity,
        "retry_count": 0,
        "sql_error": None
    }

def sql_developer_node(state: AgentState) -> AgentState:
    if con is None:
        return {"sql_query": None, "sql_data": [], "sql_error": "No DuckDB connection", "retry_count": state.get("retry_count", 0) + 1}

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
        prompt = f"Fix DuckDB SQL error.\nSCHEMA:\n{schema_info}\nFAILED QUERY: {previous_sql}\nERROR: {sql_error}\nQUESTION: {user_query}\nReturn ONLY SQL code block."
    else:
        prompt = f"""Write DuckDB SQL query.
SCHEMA:
{schema_info}
RULES:
1. Double quote column names with spaces (e.g. "Variant Name").
2. String match with ILIKE.
Return ONLY raw SQL inside ```sql ... ``` block.

QUESTION: "{user_query}"
ENTITIES: {target_entities}
"""

    response = client.chat.completions.create(
        model="openai/gpt-oss-20b",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0
    )

    raw_text = response.choices[0].message.content
    sql_query = raw_text.split("```sql")[1].split("```")[0].strip() if "```sql" in raw_text else raw_text.strip()

    try:
        query_results = con.execute(sql_query).df().to_dict(orient="records")
        return {"sql_query": sql_query, "sql_data": query_results, "sql_error": None, "retry_count": retry_count}
    except Exception as e:
        return {"sql_query": sql_query, "sql_data": None, "sql_error": str(e), "retry_count": retry_count + 1}

def vector_search_node(state: AgentState) -> AgentState:
    user_query = state["user_query"]
    target_entities = state.get("target_entity", [])

    query_vector = embedder.encode([user_query]).tolist()
    where_filter = {"Variant Name": {"$in": target_entities}} if target_entities else None
    formatted_docs = []

    try:
        review_results = collection_reviews.query(query_embeddings=query_vector, n_results=3, where=where_filter)
        rev_docs = review_results.get("documents", [[]])[0]
        rev_metas = review_results.get("metadatas", [[]])[0]
        for doc, meta in zip(rev_docs, rev_metas):
            variant = meta.get("Variant Name", "Unknown")
            formatted_docs.append(f"💬 [USER REVIEW - {variant}]: {doc}")
    except Exception:
        pass

    try:
        feature_results = collection_feature.query(query_embeddings=query_vector, n_results=3, where=where_filter)
        feat_docs = feature_results.get("documents", [[]])[0]
        feat_metas = feature_results.get("metadatas", [[]])[0]
        for doc, meta in zip(feat_docs, feat_metas):
            variant = meta.get("Variant Name", "Unknown")
            formatted_docs.append(f"🛠️ [FEATURE SPEC - {variant}]: {doc}")
    except Exception:
        pass

    if not formatted_docs:
        fallback = collection_reviews.query(query_embeddings=query_vector, n_results=4)
        formatted_docs = fallback.get("documents", [[]])[0]

    return {"vector_docs": formatted_docs}

def synthesizer_node(state):
    # 1. Safely extract variables from state with default fallbacks
    user_query = state.get("user_query", "")
    sql_data = state.get("sql_data", None)
    
    # Check for state keys (adjust 'vector_data' if your graph state uses 'vector_results')
    vector_data = state.get("vector_data") or state.get("vector_results") or ""

    # 2. Format the prompt cleanly
    synthesizer_prompt = f"""
You are a factual bike assistant. Answer the user prompt strictly using ONLY the provided context below.

SQL Data Context:
{sql_data if sql_data is not None else "No database records found."}

Vector Review Context:
{vector_data if vector_data else "No relevant reviews found."}

User Query: {user_query}

STRICT GUIDELINES:
1. If the context is empty, null, or does not contain sufficient details to answer the query, state: "I'm sorry, but I don't have enough data in my records to answer that question."
2. Do NOT guess, fabricate, or rely on pre-trained external knowledge.
"""

    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": synthesizer_prompt}],
        temperature=0.0
    )

    return {"final_response": response.choices[0].message.content}

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

app = workflow.compile()


# --- 6. STREAMLIT INTERFACE ---

st.set_page_config(page_title="Bike AI Assistant", page_icon="🏍️", layout="wide")
st.title("🏍️ Multi-Agent Bike Assistant")

with st.sidebar:
    st.header("⚙️ Configuration")
    key_input = st.text_input("Groq API Key", type="password", value=st.secrets.get("GROQ_API_KEY", os.environ.get("GROQ_API_KEY", "")))
    if key_input:
        os.environ["GROQ_API_KEY"] = key_input

if "messages" not in st.session_state:
    st.session_state.messages = []

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
                out = app.invoke({"user_query": prompt})
                status.update(label="Complete", state="complete", expanded=False)
                res_text = out.get("final_response", "No response generated.")

        st.markdown(res_text)
        st.session_state.messages.append({"role": "assistant", "content": res_text})
