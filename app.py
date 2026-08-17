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

# --- 1. STATE & SCHEMA DEFINITIONS ---

class AgentState(TypedDict):
    user_query: str
    route: str
    target_entity: List[str]
    sql_query: Optional[str]
    sql_data: Optional[List[dict]]
    vector_data: Optional[List[str]]  # Fixed key name consistency
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
        st.write(f"⚠️ Retrying SQL Node (Attempt {retry_count}/3)")
        return "sql_developer"
    return "synthesizer"


# --- 4. NODE DEFINITIONS ---

def supervisor_node(state: AgentState) -> AgentState:
    if client is None:
        return {"route": "VECTOR", "target_entity": [], "retry_count": 0, "sql_error": None}

    user_query = state["user_query"]

    prompt = """
You are a routing supervisor for a bike information assistant. Analyze the user query and return ONLY a JSON object matching this schema:

{
  "route": "SQL" | "VECTOR" | "HYBRID",
  "target_entity": ["bike model or brand names"],
  "reasoning": "explanation for routing choice"
}

ROUTING RULES:
1. Choose "SQL" ONLY if the query asks purely for structured specifications, prices, mileage, or technical features.
2. Choose "VECTOR" ONLY if the query asks purely for user reviews, opinions, pros/cons, or feedback.
3. Choose "HYBRID" if the query asks for BOTH features/specs AND reviews/opinions together, OR if it asks a general question comparing both aspects.

EXAMPLES:
- "Honda CB350 features" -> "SQL"
- "Honda CB350 review" -> "VECTOR"
- "Honda CB350 features and review" -> "HYBRID"

IMPORTANT:
- Output "route" as strictly one of: "SQL", "VECTOR", or "HYBRID".
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
    Columns: Varient_Name (Note exact spelling with 'e'), Average_stars, Review_title, User_rating, Review_description
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
    sql_query = raw_text.split("```sql")[1].split("
