import os
import sqlite3
import json
from typing import Any, List
from fastapi import FastAPI, HTTPException, Header
from pydantic import BaseModel
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.runnables import RunnableLambda
import uvicorn
import requests


# Load environment variables from .env file
load_dotenv()

# --- Database setup ---
DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "properties.db")

def _init_db():
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS unverified_prop (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                intent TEXT,
                landmark_details TEXT,
                property_type TEXT,
                rooms INTEGER,
                price INTEGER,
                latitude TEXT,
                longitude TEXT,
                amenities TEXT,
                owner_name TEXT,
                owner_mobile_number INTEGER,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        conn.commit()

_init_db()

def _save_to_unverified_prop(payload: dict[str, Any]) -> int:
    amenities = payload.get("amenities")
    amenities_str = json.dumps(amenities) if amenities is not None else None
    with sqlite3.connect(DB_PATH) as conn:
        cursor = conn.execute("""
            INSERT INTO unverified_prop
                (intent, landmark_details, property_type, rooms, price,
                 latitude, longitude, amenities, owner_name, owner_mobile_number)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            payload.get("intent"),
            payload.get("landmark_details"),
            payload.get("property_type"),
            payload.get("rooms"),
            payload.get("price"),
            payload.get("latitude"),
            payload.get("longitude"),
            amenities_str,
            payload.get("owner_name"),
            payload.get("owner_mobile_number"),
        ))
        conn.commit()
        return int(cursor.lastrowid)


def _extract_sql_from_response(raw: str) -> str:
    text = raw.strip()
    if "```" in text:
        chunks = [c.strip() for c in text.split("```") if c.strip()]
        if chunks:
            text = chunks[0]
            if text.lower().startswith("sql"):
                text = text[3:].strip()
    return text.strip().strip(";")


def _validate_rent_sql(sql: str) -> str:
    cleaned = " ".join(sql.strip().split())
    lowered = cleaned.lower()
    if not lowered.startswith("select"):
        raise ValueError("Generated SQL must start with SELECT")
    if " from property" not in lowered:
        raise ValueError("Generated SQL must query PROPERTY table")

    blocked_terms = [";", "--", "/*", "*/", "insert", "update", "delete", "drop", "alter", "create", "pragma", "attach"]
    for term in blocked_terms:
        if term in lowered:
            raise ValueError("Unsafe SQL generated")
    return cleaned


def _generate_rent_sql_query(user_message: str) -> str:
    sql_prompt = f"""
You are an SQL generator for SQLite.
Generate exactly one SQL SELECT query for the PROPERTY table.

Table schema:
- id INTEGER
- landmark TEXT
- property_type TEXT
- rooms INTEGER
- rent INTEGER
- latitude TEXT
- longitude TEXT
- amenities TEXT

Rules:
- Output only SQL (no markdown, no explanation).
- Always start with SELECT * FROM PROPERTY.
- If location is provided, use: WHERE lower(landmark) = lower('<location>').
- If budget/rent cap is provided, include: rent < <value>.
- Combine conditions with AND.

User message: {user_message}
"""
    response = llm.invoke(sql_prompt)
    raw = response.content if hasattr(response, "content") else str(response)
    return _validate_rent_sql(_extract_sql_from_response(raw))


def _execute_property_query(sql_query: str) -> list[dict[str, Any]]:
    with sqlite3.connect(DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        table_exists = conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND lower(name)='property' LIMIT 1"
        ).fetchone()
        if not table_exists:
            raise ValueError("PROPERTY table not found in properties.db")
        rows = conn.execute(sql_query).fetchall()
        return [dict(row) for row in rows]


@tool
def get_overpass_insights(latitude: str, longitude: str, limit: int = 5) -> dict:
    """
    Fetches nearby famous landmarks and top amenities from Overpass around coordinates.
    """
    if not latitude or not longitude:
        return {"famous_landmarks": [], "top_amenities": []}

    overpass_urls = [
        "https://overpass.kumi.systems/api/interpreter",
        "https://overpass-api.de/api/interpreter",
        "https://overpass.openstreetmap.ru/api/interpreter",
    ]
    rich_query = f"""
    [out:json][timeout:25];
    (
      node(around:2000,{latitude},{longitude})["tourism"];
      node(around:2000,{latitude},{longitude})["historic"];
      node(around:2000,{latitude},{longitude})["amenity"];
      way(around:2000,{latitude},{longitude})["tourism"];
      way(around:2000,{latitude},{longitude})["historic"];
      way(around:2000,{latitude},{longitude})["amenity"];
    );
    out tags center 100;
    """

    # Fallback query is smaller and tends to succeed more often on busy endpoints.
    amenity_only_query = f"""
    [out:json][timeout:25];
    (
      node(around:1500,{latitude},{longitude})["amenity"];
      way(around:1500,{latitude},{longitude})["amenity"];
    );
    out tags center 120;
    """

    data = None
    for query in [rich_query, amenity_only_query]:
        for overpass_url in overpass_urls:
            for _attempt in range(2):
                try:
                    response = requests.post(
                        overpass_url,
                        data={"data": query},
                        timeout=20,
                        headers={"User-Agent": "AG_AI-FastAPI/1.0"},
                    )
                except requests.exceptions.RequestException:
                    continue

                if not response.ok:
                    continue

                try:
                    candidate = response.json()
                except ValueError:
                    continue

                if candidate.get("elements"):
                    data = candidate
                    break

            if data is not None:
                break
        if data is not None:
            break

    if data is None:
        return {"famous_landmarks": [], "top_amenities": []}

    landmarks: list[str] = []
    amenity_details: dict[str, dict[str, Any]] = {}

    for element in data.get("elements", []):
        tags = element.get("tags", {})
        name = tags.get("name")
        if ("tourism" in tags or "historic" in tags) and name and name not in landmarks:
            landmarks.append(name)

        amenity_value = tags.get("amenity")
        if amenity_value:
            if amenity_value not in amenity_details:
                amenity_details[amenity_value] = {
                    "count": 0,
                    "places": []
                }
            amenity_details[amenity_value]["count"] += 1

            if name and name not in amenity_details[amenity_value]["places"]:
                amenity_details[amenity_value]["places"].append(name)

    sorted_amenities = sorted(
        amenity_details.items(),
        key=lambda kv: kv[1]["count"],
        reverse=True,
    )[:limit]

    top_amenities: list[dict[str, Any]] = []
    for amenity_type, details in sorted_amenities:
        top_amenities.append({
            "amenity_type": amenity_type,
            "total_nearby": details["count"],
            "places": details["places"][:5],
        })

    return {
        "famous_landmarks": landmarks[:limit],
        "top_amenities": top_amenities,
    }


def _generate_amenities_feedback(amenities_object: dict[str, Any]) -> str:
    feedback_prompt = f"""
Generate a high level feedback by processing these amenities.
Keep it concise (1-2 sentences), useful for a renter.

Data:
{json.dumps(amenities_object, indent=2)}
"""
    try:
        response = llm.invoke(feedback_prompt)
        return response.content if hasattr(response, "content") else str(response)
    except Exception:
        return "This property appears to be in a reasonably serviced area with useful nearby amenities."


def _generate_no_properties_message(user_message: str, sql_query: str | None = None) -> str:
    prompt = f"""
You are a rental assistant.
The user searched for rental properties, but no matching properties were found.
Write a short, polite, helpful response in 1-2 sentences.
Suggest that the user can relax location/budget filters or try another area.

User message: {user_message}
SQL used: {sql_query}
"""
    try:
        response = llm.invoke(prompt)
        return response.content if hasattr(response, "content") else str(response)
    except Exception:
        return "No matching properties were found for your request. Please try a nearby location or increase your budget range."


def _rent_prepare_context(inputs: dict[str, Any]) -> dict[str, Any]:
    return {"message": inputs["message"]}


def _rent_attach_sql(inputs: dict[str, Any]) -> dict[str, Any]:
    sql_query = _generate_rent_sql_query(inputs["message"])
    return {**inputs, "sql_query": sql_query}


def _rent_execute_sql(inputs: dict[str, Any]) -> dict[str, Any]:
    properties = _execute_property_query(inputs["sql_query"])
    return {**inputs, "properties": properties}


def _rent_enrich_with_overpass(inputs: dict[str, Any]) -> dict[str, Any]:
    enriched: list[dict[str, Any]] = []
    for idx, prop in enumerate(inputs.get("properties", []), start=1):
        latitude = prop.get("latitude")
        longitude = prop.get("longitude")
        insights = get_overpass_insights.invoke({
            "latitude": str(latitude) if latitude is not None else "",
            "longitude": str(longitude) if longitude is not None else "",
            "limit": 5,
        })

        amenities_obj = {
            "property_id": prop.get("id"),
            "landmark": prop.get("landmark"),
            "top_amenities": insights.get("top_amenities", []),
        }
        feedback = _generate_amenities_feedback(amenities_obj)

        enriched.append({
            "display_order": idx,
            "property": prop,
            "famous_landmarks": insights.get("famous_landmarks", []),
            "top_amenities": insights.get("top_amenities", []),
            "amenities_feedback": feedback,
        })

    return {**inputs, "enriched_properties": enriched}


def _rent_build_response(inputs: dict[str, Any]) -> dict[str, Any]:
    properties = inputs.get("enriched_properties", [])
    if not properties:
        no_results_message = _generate_no_properties_message(
            inputs.get("message", ""),
            inputs.get("sql_query")
        )
        return {
            "intent": "rent_property",
            "sql_query": inputs.get("sql_query"),
            "results_count": 0,
            "properties": [],
            "message": no_results_message,
        }

    return {
        "intent": "rent_property",
        "sql_query": inputs.get("sql_query"),
        "results_count": len(properties),
        "properties": properties,
        "message": "Here are the available properties, shown one by one with nearby famous landmarks and amenity-based feedback.",
    }


rent_property_flow = (
    RunnableLambda(_rent_prepare_context)
    | RunnableLambda(_rent_attach_sql)
    | RunnableLambda(_rent_execute_sql)
    | RunnableLambda(_rent_enrich_with_overpass)
    | RunnableLambda(_rent_build_response)
)

def initialize_llm():
    """Initialize and return a ChatOpenAI (Gemini) model instance."""
    # Get API details from environment variables
    api_key = os.getenv("GEMINI_API_KEY")
    model_name = os.getenv("GEMINI_MODEL_NAME")
    base_url = os.getenv("GEMINI_BASE_URL")

    if not api_key:
        raise ValueError("GEMINI_API_KEY environment variable is required. Please set it in your .env file.")
    
    if not base_url:
        raise ValueError("GEMINI_BASE_URL environment variable is required. Please set it in your .env file.")
    
    if not model_name:
        raise ValueError("GEMINI_MODEL_NAME environment variable is required. Please set it in your .env file.")
    
    # Initialize and return ChatOpenAI instance
    return ChatOpenAI(
        base_url=base_url,
        model=model_name,
        api_key=api_key
    )

# Create FastAPI app
app = FastAPI(title="LangChain + Gemini Demo", version="1.0.0")
# Initialize model
llm = initialize_llm()

# Short-term memory: global conversation history (per-session in production use a DB)
conversation_history: List = []
extraction_history: List[dict[str, Any]] = []
hosted_properties: List[dict[str, Any]] = []

REQUIRED_DETAILS = [
    "intent",
    "landmark_details",
    "property_type",
    "rooms",
    "price",
    "amenities",
]

HOST_REQUIRED_DETAILS = [
    "owner_name",
    "owner_mobile_number",
]


class ChatRequest(BaseModel):
    message: str


class ChatResponse(BaseModel):
    response: dict[str, Any]
    model: str


class PropertyExtraction(BaseModel):
    intent: str | None = None
    landmark_details: str | None = None
    property_type: str | None = None
    rooms: int | None = None
    price: int | None = None
    latitude: str | None = None
    longitude: str | None = None
    amenities: list[str] | None = None
    owner_name: str | None = None
    owner_mobile_number: int | None = None


def _is_follow_up_question(message: str) -> bool:
    lowered = message.lower()
    follow_up_markers = [
        "where",
        "what",
        "which",
        "earlier",
        "before",
        "previous",
        "that",
        "it",
        "my property"
    ]
    return any(marker in lowered for marker in follow_up_markers)


def _has_any_property_details(payload: dict[str, Any]) -> bool:
    detail_fields = [
        "landmark_details",
        "property_type",
        "rooms",
        "price",
        "latitude",
        "longitude",
        "amenities",
        "owner_name",
        "owner_mobile_number",
    ]
    for field in detail_fields:
        value = payload.get(field)
        if value is None:
            continue
        if isinstance(value, str) and not value.strip():
            continue
        if isinstance(value, list) and len(value) == 0:
            continue
        return True
    return False


def _is_new_property_request(message: str) -> bool:
    lowered = message.lower()
    new_markers = [
        "another property",
        "new property",
        "second property",
        "next property",
    ]
    return any(marker in lowered for marker in new_markers)


def _is_property_info_question(message: str) -> bool:
    classification_prompt = f"""You are an intent classifier.
Determine if the user's message is asking about details of a property they have already provided
(e.g. asking about the landmark, price, rooms, amenities, property type, owner name, owner mobile number, or any other previously submitted property detail).

Reply with ONLY the single word "yes" or "no". No punctuation, no explanation.

User message: "{message}"
"""
    try:
        response = llm.invoke(classification_prompt)
        answer = (response.content if hasattr(response, "content") else str(response)).strip().lower()
        return answer.startswith("yes")
    except Exception:
        return False


def _merge_with_memory(current: dict[str, Any], user_message: str) -> dict[str, Any]:
    if not extraction_history:
        return current

    should_merge = _is_follow_up_question(user_message)
    if not should_merge:
        # Also merge when user sends detail-only updates like "at Visakhapatnam".
        should_merge = current.get("intent") is None and _has_any_property_details(current)

    if not should_merge:
        return current

    previous = extraction_history[-1]
    merged = dict(current)
    for key, value in merged.items():
        if value is None and previous.get(key) is not None:
            merged[key] = previous[key]
    return merged


def _missing_required_details(payload: dict[str, Any]) -> list[str]:
    missing: list[str] = []
    required_fields = list(REQUIRED_DETAILS)
    if payload.get("intent") == "host_property":
        required_fields.extend(HOST_REQUIRED_DETAILS)

    for field in required_fields:
        value = payload.get(field)
        if value is None:
            missing.append(field)
            continue
        if isinstance(value, str) and not value.strip():
            missing.append(field)
            continue
        if isinstance(value, list) and len(value) == 0:
            missing.append(field)
    return missing


def _build_missing_details_message(missing: list[str]) -> str:
    labels = {
        "intent": "intent (host_property or rent_property)",
        "landmark_details": "location/landmark",
        "property_type": "property type",
        "rooms": "number of rooms",
        "price": "price",
        "amenities": "amenities",
        "owner_name": "owner name",
        "owner_mobile_number": "owner mobile number",
    }
    missing_labels = [labels.get(item, item) for item in missing]
    return (
        "Thanks for the details so far. To proceed further, please provide: "
        + ", ".join(missing_labels)
        + "."
    )


def _build_host_success_message(payload: dict[str, Any]) -> str:
    success_prompt = f"""
You are a helpful assistant.
The user has successfully provided all required details to host a property.
Return one short positive confirmation message.

Property details:
{payload}

Example style:
Thank you for listing the property. It will be displayed to customers once the admin approves it.
"""
    try:
        response = llm.invoke(success_prompt)
        return response.content if hasattr(response, "content") else str(response)
    except Exception:
        return "Thank you for listing the property. It will be displayed to customers once the admin approves it."


def _build_property_info_message(user_message: str, payload: dict[str, Any]) -> str:
    lowered = user_message.lower()
    landmark = payload.get("landmark_details")
    property_type = payload.get("property_type")
    rooms = payload.get("rooms")
    price = payload.get("price")
    amenities = payload.get("amenities")
    intent = payload.get("intent")
    owner_name = payload.get("owner_name")
    owner_mobile_number = payload.get("owner_mobile_number")

    if "landmark" in lowered and landmark:
        return (
            f"Dear customer, the landmark of your property is {landmark}. "
            "If you need more details, please let me know."
        )

    if ("price" in lowered or "rent" in lowered) and price is not None:
        return (
            f"Dear customer, the price of your property is {price}. "
            "If you need more details, please let me know."
        )

    if "room" in lowered and rooms is not None:
        return (
            f"Dear customer, your property has {rooms} rooms. "
            "If you need more details, please let me know."
        )

    if ("amenities" in lowered or "amenity" in lowered) and amenities:
        amenities_text = ", ".join(amenities)
        return (
            f"Dear customer, the amenities of your property are {amenities_text}. "
            "If you need more details, please let me know."
        )

    if ("type" in lowered or "property type" in lowered) and property_type:
        return (
            f"Dear customer, the property type is {property_type}. "
            "If you need more details, please let me know."
        )

    if "intent" in lowered and intent:
        return (
            f"Dear customer, your property request intent is {intent}. "
            "If you need more details, please let me know."
        )

    if ("owner name" in lowered or "name of owner" in lowered) and owner_name:
        return (
            f"Dear customer, the owner name is {owner_name}. "
            "If you need more details, please let me know."
        )

    if ("owner mobile" in lowered or "owner number" in lowered or "mobile number" in lowered) and owner_mobile_number:
        return (
            f"Dear customer, the owner mobile number is {owner_mobile_number}. "
            "If you need more details, please let me know."
        )

    return "Dear customer, I can help with your saved property details. Please ask what detail you need."

# --- Step 4: Define Nominatim Tool ---
@tool
def get_coordinates(place: str) -> dict:
    """
    Fetches latitude and longitude coordinates for a given place name using Nominatim.
    """
    if not place:
        return {"latitude": None, "longitude": None, "display_name": None}

    url = "https://nominatim.openstreetmap.org/search"
    params = {"q": place, "format": "json", "limit": 1}
    try:
        response = requests.get(url, params=params, headers={"User-Agent": "FastAPI-App"}, timeout=5)
        response.raise_for_status()
        data = response.json()
        if not data:
            return {"latitude": None, "longitude": None, "display_name": None}
        location = data[0]
        return {
            "latitude": location.get("lat"),
            "longitude": location.get("lon"),
            "display_name": location.get("display_name")
        }
    except requests.exceptions.RequestException:
        return {"latitude": None, "longitude": None, "display_name": None}

@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest) -> ChatResponse:
    """
    Basic LLM invocation endpoint demonstrating the fundamental workflow:
    
    1. Environment Setup: GEMINI_API_KEY, GEMINI_MODEL_NAME, and GEMINI_BASE_URL loaded from .env file
    2. Instantiation: ChatOpenAI instance created with model from GEMINI_MODEL_NAME environment variable
    3. Invocation: Call the .invoke() method with the user's message
    4. Response Handling: LangChain sends request to Gemini API and parses JSON response into AIMessage
    5. Output: Return the content of the AIMessage object
    """
    try:
        prompt = """
You are an information extraction assistant. 
Your task is to analyze the user's message about property hosting or renting and return the details in strict JSON format.

Instructions:
- Identify the user's intent: either "host_property" or "rent_property".
- Extract landmark details (e.g., neighborhood, city, nearby places).
- Extract additional property details if mentioned (e.g., property type, number of rooms, price, amenities).
- If intent is host_property, also extract owner_name and owner_mobile_number.
- If a detail is not provided, return null for that field.
- Do not add extra commentary, only return JSON.
- Also, other than latitude and longitude if any other details are missing, return a polite message indicating which detail is missing.

Output JSON schema:
{
  "intent": "host_property | rent_property | null",
  "landmark_details": "string | null",
  "property_type": "string | null",
  "rooms": "integer | null",
  "price": "integer | null",
  "latitude": "string | null",
  "longitude": "string | null",
    "amenities": ["string"] | null,
    "owner_name": "string | null",
    "owner_mobile_number": "integer | null"
}

Example:
User: "I want to host my 2BHK apartment near Maddilapalem with parking available."
Response:
{
  "intent": "host_property",
  "landmark_details": "Maddilapalem",
  "property_type": "apartment",
  "rooms": 2,
  "price": null,
  "latitude": null,
  "longitude": null,
    "amenities": ["parking"],
    "owner_name": null,
    "owner_mobile_number": null
}
"""

        structured_llm = llm.with_structured_output(PropertyExtraction)
        chain = ChatPromptTemplate.from_messages([
            ("system", "{instruction}"),
            MessagesPlaceholder(variable_name="chat_history"),
            ("user", "{message}")
        ]) | structured_llm
        history_for_llm = [] if _is_new_property_request(request.message) else conversation_history

        extraction = chain.invoke({
            "instruction": prompt,
            "message": request.message,
            "chat_history": history_for_llm
        })

        landmark = extraction.landmark_details
        coords = get_coordinates.invoke({"place": landmark}) if landmark else {"latitude": None, "longitude": None}

        if extraction.latitude is None:
            extraction.latitude = coords.get("latitude")
        if extraction.longitude is None:
            extraction.longitude = coords.get("longitude")

        # Keep response aligned with the requested output schema.
        response_payload = {
            "intent": extraction.intent,
            "landmark_details": extraction.landmark_details,
            "property_type": extraction.property_type,
            "rooms": extraction.rooms,
            "price": extraction.price,
            "latitude": extraction.latitude,
            "longitude": extraction.longitude,
            "amenities": extraction.amenities,
            "owner_name": extraction.owner_name,
            "owner_mobile_number": extraction.owner_mobile_number
        }

        if _is_new_property_request(request.message):
            resolved_payload = dict(response_payload)
        else:
            resolved_payload = _merge_with_memory(response_payload, request.message)

        if _is_property_info_question(request.message):
            source_payload = extraction_history[-1] if extraction_history else (hosted_properties[-1] if hosted_properties else resolved_payload)
            info_message = _build_property_info_message(request.message, source_payload)
            response_info = {"message": info_message}
            conversation_history.append(HumanMessage(content=request.message))
            conversation_history.append(AIMessage(content=info_message))
            model_name = os.getenv("GEMINI_MODEL_NAME", "unknown")
            return ChatResponse(response=response_info, model=model_name)

        if resolved_payload.get("intent") == "rent_property":
            rent_response = rent_property_flow.invoke({"message": request.message})
            conversation_history.append(HumanMessage(content=request.message))
            conversation_history.append(AIMessage(content=str(rent_response)))
            extraction_history.append(resolved_payload)
            model_name = os.getenv("GEMINI_MODEL_NAME", "unknown")
            return ChatResponse(response=rent_response, model=model_name)

        missing_fields = _missing_required_details(resolved_payload)
        if missing_fields:
            polite_message = _build_missing_details_message(missing_fields)
            response_with_prompt = dict(resolved_payload)
            response_with_prompt["message"] = polite_message
            conversation_history.append(HumanMessage(content=request.message))
            conversation_history.append(AIMessage(content=str(response_with_prompt)))
            extraction_history.append(resolved_payload)
            model_name = os.getenv("GEMINI_MODEL_NAME", "unknown")
            return ChatResponse(response=response_with_prompt, model=model_name)

        final_payload = dict(resolved_payload)
        if resolved_payload.get("intent") == "host_property":
            final_payload["message"] = _build_host_success_message(final_payload)
            hosted_properties.append(dict(final_payload))
            _save_to_unverified_prop(resolved_payload)

        conversation_history.append(HumanMessage(content=request.message))
        conversation_history.append(AIMessage(content=str(final_payload)))
        extraction_history.append(resolved_payload)

        model_name = os.getenv("GEMINI_MODEL_NAME", "unknown")
        return ChatResponse(response=final_payload, model=model_name)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


# ---------------------------------------------------------------------------
# Admin helpers (LCEL-powered approval flow)
# ---------------------------------------------------------------------------

def _require_admin(x_admin_key: str | None) -> None:
    """Raise 403 if the provided key doesn't match ADMIN_KEY in .env."""
    expected = os.getenv("ADMIN_KEY", "")
    if not expected:
        raise HTTPException(status_code=500, detail="ADMIN_KEY is not configured in .env")
    if x_admin_key != expected:
        raise HTTPException(status_code=403, detail="Invalid or missing X-Admin-Key header")


# --- LCEL steps ---

def _admin_fetch_record(inputs: dict[str, Any]) -> dict[str, Any]:
    """Fetch a single row from unverified_prop by id."""
    record_id = inputs["id"]
    with sqlite3.connect(DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        row = conn.execute(
            "SELECT * FROM unverified_prop WHERE id = ?", (record_id,)
        ).fetchone()
    if not row:
        raise ValueError(f"No pending property found with id={record_id}")
    return {**inputs, "record": dict(row)}


def _admin_insert_to_property(inputs: dict[str, Any]) -> dict[str, Any]:
    """Insert the approved record into the PROPERTY table."""
    rec = inputs["record"]
    amenities = rec.get("amenities")  # already a JSON string in DB
    with sqlite3.connect(DB_PATH) as conn:
        cursor = conn.execute(
            """
            INSERT INTO PROPERTY (
                landmark,
                property_type,
                rooms,
                rent,
                latitude,
                longitude,
                amenities,
                owner_name,
                owner_mobile_number,
                source_unverified_id
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                rec.get("landmark_details"),
                rec.get("property_type"),
                rec.get("rooms"),
                rec.get("price"),
                rec.get("latitude"),
                rec.get("longitude"),
                amenities,
                rec.get("owner_name"),
                rec.get("owner_mobile_number"),
                rec.get("id"),
            ),
        )
        conn.commit()
        new_id = int(cursor.lastrowid)
    return {**inputs, "new_property_id": new_id}


def _admin_delete_from_unverified(inputs: dict[str, Any]) -> dict[str, Any]:
    """Remove the now-approved entry from unverified_prop."""
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute("DELETE FROM unverified_prop WHERE id = ?", (inputs["id"],))
        conn.commit()
    return inputs


def _admin_build_approval_response(inputs: dict[str, Any]) -> dict[str, Any]:
    rec = inputs["record"]
    return {
        "status": "approved",
        "unverified_id": inputs["id"],
        "new_property_id": inputs["new_property_id"],
        "landmark": rec.get("landmark_details"),
        "property_type": rec.get("property_type"),
        "rooms": rec.get("rooms"),
        "rent": rec.get("price"),
        "message": f"Property '{rec.get('landmark_details')}' has been approved and is now live.",
    }


approve_property_flow = (
    RunnableLambda(_admin_fetch_record)
    | RunnableLambda(_admin_insert_to_property)
    | RunnableLambda(_admin_delete_from_unverified)
    | RunnableLambda(_admin_build_approval_response)
)


# --- Admin endpoints ---

@app.get("/admin/pending")
def admin_list_pending(x_admin_key: str | None = Header(default=None)) -> dict[str, Any]:
    """List all properties waiting for approval."""
    _require_admin(x_admin_key)
    with sqlite3.connect(DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute("SELECT * FROM unverified_prop ORDER BY created_at DESC").fetchall()
    return {"pending_count": len(rows), "properties": [dict(r) for r in rows]}


@app.post("/admin/approve/{property_id}")
def admin_approve(property_id: int, x_admin_key: str | None = Header(default=None)) -> dict[str, Any]:
    """Approve a pending property: move it from unverified_prop to PROPERTY."""
    _require_admin(x_admin_key)
    try:
        result = approve_property_flow.invoke({"id": property_id})
        return result
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Approval failed: {exc}")


@app.post("/admin/reject/{property_id}")
def admin_reject(property_id: int, x_admin_key: str | None = Header(default=None)) -> dict[str, Any]:
    """Reject and permanently delete a pending property."""
    _require_admin(x_admin_key)
    with sqlite3.connect(DB_PATH) as conn:
        row = conn.execute("SELECT id FROM unverified_prop WHERE id = ?", (property_id,)).fetchone()
        if not row:
            raise HTTPException(status_code=404, detail=f"No pending property with id={property_id}")
        conn.execute("DELETE FROM unverified_prop WHERE id = ?", (property_id,))
        conn.commit()
    return {"status": "rejected", "unverified_id": property_id, "message": "Property has been rejected and removed."}


if __name__ == "__main__":
    app_port = int(os.getenv("APP_PORT", "8011"))
    uvicorn.run(app, host="0.0.0.0", port=app_port)
