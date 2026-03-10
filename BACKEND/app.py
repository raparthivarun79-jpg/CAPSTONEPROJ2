from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
import math
import json
import re
from sqlalchemy import create_engine, Column, Integer, String, Float, Boolean
from sqlalchemy.orm import declarative_base, sessionmaker
from sqlalchemy.exc import SQLAlchemyError
import requests
import os
import sqlite3
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
import time
from langchain_core.tools import tool

# Initialize FastAPI app
app = FastAPI()
load_dotenv()
# Database setup
DATABASE_URL = "sqlite:///./real_estate.db"
engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
Base = declarative_base()
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def ensure_properties_schema_compatibility() -> None:
    """Make legacy and new column names coexist in SQLite for safer upgrades."""
    conn = sqlite3.connect("real_estate.db")
    cur = conn.cursor()
    cur.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='properties'")
    table_exists = cur.fetchone() is not None
    if not table_exists:
        conn.close()
        return

    cur.execute("PRAGMA table_info(properties)")
    columns = {row[1] for row in cur.fetchall()}

    if "owner_name" not in columns and "name" in columns:
        cur.execute("ALTER TABLE properties ADD COLUMN owner_name VARCHAR")
        cur.execute("UPDATE properties SET owner_name = name WHERE owner_name IS NULL")

    if "rent_per_month" not in columns and "rent" in columns:
        cur.execute("ALTER TABLE properties ADD COLUMN rent_per_month FLOAT")
        cur.execute("UPDATE properties SET rent_per_month = rent WHERE rent_per_month IS NULL")

    if "max_occupants" not in columns and "max_people" in columns:
        cur.execute("ALTER TABLE properties ADD COLUMN max_occupants INTEGER")
        cur.execute("UPDATE properties SET max_occupants = max_people WHERE max_occupants IS NULL")

    conn.commit()
    conn.close()

# Define database models
class Property(Base):
    __tablename__ = "properties"

    id = Column(Integer, primary_key=True, index=True)
    # Keep Python attribute names stable while matching existing DB column names.
    owner_name = Column("name", String, index=True)
    rent_per_month = Column("rent", Float)
    max_occupants = Column("max_people", Integer)
    property_type = Column(String)
    floor = Column(String)
    pet_friendly = Column(Boolean)
    latitude = Column(Float)
    longitude = Column(Float)

Base.metadata.create_all(bind=engine)
ensure_properties_schema_compatibility()

# Pydantic models for request/response validation
class HostPropertyRequest(BaseModel):
    owner_name: str
    rent_per_month: float
    max_occupants: int
    property_type: str
    floor: str
    pet_friendly: bool
    address: str

class RentPropertyRequest(BaseModel):
    landmark: str
    range_km: float


class AmenityResult(BaseModel):
    source: str | None = None
    category: str | None = None
    title: str | None = None
    rating: float | None = None
    reviews: float | None = None
    address: str | None = None
    phone: str | None = None
    website: str | None = None
    direction: str | None = None
    latitude: float | None = None
    longitude: float | None = None
    distance_from_property_km: float | None = None


class LandmarkResult(BaseModel):
    title: str
    category: str | None = None
    distance_km: float


class PropertyResult(BaseModel):
    id: int
    owner_name: str | None = None
    rent_per_month: float | None = None
    latitude: float
    longitude: float
    distance_km: float
    amenities: List[AmenityResult] = []
    nearby_landmarks: List[LandmarkResult] = []


class RentPropertyResponse(BaseModel):
    properties: List[PropertyResult]
    amenities: List[AmenityResult]
    nearby_landmarks: List[LandmarkResult]
    summary: str


class AgentChatRequest(BaseModel):
    message: str
    session_id: str | None = "default"


class AgentChatResponse(BaseModel):
    intent: str
    selected_endpoint: str
    extracted_payload: dict
    result: dict

# OpenStreetMap API integration
NOMINATIM_URL = "https://nominatim.openstreetmap.org/search"

# Define Overpass API URL
OVERPASS_URL = "https://overpass-api.de/api/interpreter"
OVERPASS_URLS = [
    "https://overpass-api.de/api/interpreter",
    "https://overpass.kumi.systems/api/interpreter",
    "https://lz4.overpass-api.de/api/interpreter",
]
SEARCHAPI_URL = "https://www.searchapi.io/api/v1/search"
OSRM_ROUTE_URL = "https://router.project-osrm.org/route/v1/driving"
MAX_AMENITIES_RESULTS = 40
MAX_PROPERTIES_FOR_ROUTING = 25
OVERPASS_NODE_LIMIT = 120
CACHE_TTL_SECONDS = 300
MAX_LANDMARK_RESULTS = 5

_geocode_cache: dict[str, tuple[float, dict]] = {}
_amenities_cache: dict[str, tuple[float, list[dict]]] = {}
_llm_retry_after_ts = 0.0
_agent_conversation_history: dict[str, list] = {}
MAX_CHAT_HISTORY_MESSAGES = 12

# Initialize the LLM
def initialize_llm():
    """Initialize and return a ChatOpenAI model instance."""
    api_key = os.getenv("GEMINI_API_KEY")
    model_name = os.getenv("GEMINI_MODEL_NAME")
    base_url = os.getenv("GEMINI_BASE_URL")

    if not api_key:
        raise ValueError("GEMINI_API_KEY environment variable is required. Please set it in your .env file.")

    if not base_url:
        raise ValueError("GEMINI_BASE_URL environment variable is required. Please set it in your .env file.")

    if not model_name:
        raise ValueError("GEMINI_MODEL_NAME environment variable is required. Please set it in your .env file.")

    return ChatOpenAI(
        base_url=base_url,
        model=model_name,
        api_key=api_key
    )

llm = initialize_llm()


def safe_llm_invoke(prompt: str, fallback_text: str, timeout_seconds: int = 12) -> str:
    """Call the LLM with a timeout and return fallback text on any failure."""
    global _llm_retry_after_ts

    # Avoid adding a fixed timeout penalty to every request when provider is failing.
    if time.time() < _llm_retry_after_ts:
        return fallback_text

    try:
        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(llm.invoke, prompt)
            result = future.result(timeout=timeout_seconds)
        _llm_retry_after_ts = 0.0
        return result.content if hasattr(result, "content") else str(result)
    except (FuturesTimeoutError, Exception):
        _llm_retry_after_ts = time.time() + 120
        return fallback_text

# Define a prompt template for hosting properties
host_prompt = (
    "You are a real estate assistant. Write a short, friendly confirmation message for the property owner. "
    "Include the owner name exactly as provided. Keep it to 1-2 sentences and positive in tone. "
    "Mention that the property will be shown to interested customers and they may be contacted soon. "
    "Do not mention any errors, unavailability, or technical issues.\n"
    "Owner Name: {owner_name}\n"
    "Property Details: {property_details}"
)

# Define a prompt template for renting properties
rent_prompt = "You are an assistant helping users find rental properties. Based on the following search results:\n{search_results}\nand nearby amenities:\n{amenities}\nGenerate a natural language summary to present to the user."

agent_router_prompt = (
    "You are an API routing assistant for a real-estate app. "
    "Classify user intent and extract payload. "
    "Return ONLY strict JSON with keys: intent, host_payload, rent_payload. "
    "intent must be one of: host_property, rent_property, unknown. "
    "host_payload keys: owner_name, rent_per_month, max_occupants, property_type, floor, pet_friendly, address. "
    "rent_payload keys: landmark, range_km. "
    "Use null for missing values. "
    "For range_km, if user does not specify, set 5. "
    "Do not include markdown or explanation.\n"
    "User message: {user_message}"
)

agent_router_stateful_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are an API routing assistant for a real-estate app. "
            "Classify user intent and extract payload. "
            "Return ONLY strict JSON with keys: intent, host_payload, rent_payload. "
            "intent must be one of: host_property, rent_property, unknown. "
            "host_payload keys: owner_name, rent_per_month, max_occupants, property_type, floor, pet_friendly, address. "
            "rent_payload keys: landmark, range_km. "
            "Use null for missing values. "
            "For range_km, if user does not specify, set 5. "
            "Use chat history to resolve references like 'same as before' or omitted fields. "
            "Do not include markdown or explanation.",
        ),
        MessagesPlaceholder(variable_name="chat_history"),
        ("user", "{user_message}"),
    ]
)


def _get_session_history(session_id: str | None) -> list:
    """Return per-session short-term history buffer."""
    key = (session_id or "default").strip() or "default"
    if key not in _agent_conversation_history:
        _agent_conversation_history[key] = []
    return _agent_conversation_history[key]


def _append_session_history(session_id: str | None, user_message: str, assistant_message: str) -> None:
    """Store short-term exchange and trim to a fixed window."""
    history = _get_session_history(session_id)
    history.append(HumanMessage(content=user_message))
    history.append(AIMessage(content=assistant_message))
    if len(history) > MAX_CHAT_HISTORY_MESSAGES:
        _agent_conversation_history[(session_id or "default").strip() or "default"] = history[-MAX_CHAT_HISTORY_MESSAGES:]


def _is_memory_question(message: str) -> bool:
    """Detect user questions about prior conversation context."""
    text = (message or "").lower()
    triggers = [
        "previous",
        "earlier",
        "before",
        "last time",
        "did i ask",
        "where did i ask",
        "what did i ask",
        "what was the landmark",
    ]
    memory_terms = ["search", "rental", "rent", "host", "landmark", "property"]
    return any(t in text for t in triggers) and any(t in text for t in memory_terms)


def _build_memory_answer(session_id: str | None) -> str:
    """Return a concise memory answer from session history."""
    history = _get_session_history(session_id)
    if not history:
        return "I do not have previous chat context for this session yet."

    last_rent_landmark = None
    last_host_address = None

    for msg in reversed(history):
        if not isinstance(msg, AIMessage):
            continue
        try:
            payload = json.loads(msg.content)
        except Exception:
            continue

        intent = payload.get("intent")
        extracted = payload.get("extracted_payload") if isinstance(payload.get("extracted_payload"), dict) else {}
        if intent == "rent_property" and last_rent_landmark is None:
            landmark = extracted.get("landmark")
            if landmark:
                last_rent_landmark = landmark
        if intent == "host_property" and last_host_address is None:
            address = extracted.get("address")
            if address:
                last_host_address = address

        if last_rent_landmark and last_host_address:
            break

    if last_rent_landmark:
        return f"Previously, you asked to search rental properties near {last_rent_landmark}."
    if last_host_address:
        return f"Previously, you asked to host a property at {last_host_address}."
    return "I found session history, but could not identify a previous rent/host location."


def safe_llm_invoke_messages(messages: list, fallback_text: str, timeout_seconds: int = 8) -> str:
    """Call LLM with message list and timeout; return fallback on failure."""
    global _llm_retry_after_ts

    if time.time() < _llm_retry_after_ts:
        return fallback_text

    try:
        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(llm.invoke, messages)
            result = future.result(timeout=timeout_seconds)
        _llm_retry_after_ts = 0.0
        return result.content if hasattr(result, "content") else str(result)
    except (FuturesTimeoutError, Exception):
        _llm_retry_after_ts = time.time() + 120
        return fallback_text


def _extract_json_object(text: str) -> dict | None:
    """Parse first JSON object found in free-form model output."""
    if not text:
        return None

    stripped = text.strip()
    try:
        parsed = json.loads(stripped)
        return parsed if isinstance(parsed, dict) else None
    except json.JSONDecodeError:
        pass

    start = stripped.find("{")
    end = stripped.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None

    try:
        parsed = json.loads(stripped[start : end + 1])
        return parsed if isinstance(parsed, dict) else None
    except json.JSONDecodeError:
        return None


def _heuristic_agent_parse(message: str) -> dict:
    """Fallback intent extraction when model output is unavailable/unparseable."""
    original_text = message or ""
    text = original_text.lower()
    host_tokens = ["host", "list", "owner", "my apartment", "rent per month", "pet friendly"]
    rent_tokens = ["rent", "find", "looking for", "near", "within", "km"]

    host_score = sum(1 for token in host_tokens if token in text)
    rent_score = sum(1 for token in rent_tokens if token in text)

    if host_score >= rent_score and host_score > 0:
        intent = "host_property"
    elif rent_score > 0:
        intent = "rent_property"
    else:
        intent = "unknown"

    host_payload = {
        "owner_name": None,
        "rent_per_month": None,
        "max_occupants": None,
        "property_type": None,
        "floor": None,
        "pet_friendly": None,
        "address": None,
    }
    rent_payload = {
        "landmark": None,
        "range_km": 5,
    }

    # Extract common key=value fields from free-text messages.
    # Supports formats like owner_name='SARVANAN', rent_per_month=2000, pet_friendly=YES.
    field_map = {
        "owner_name": "owner_name",
        "rent_per_month": "rent_per_month",
        "max_occupants": "max_occupants",
        "property_type": "property_type",
        "floor": "floor",
        "pet_friendly": "pet_friendly",
        "address": "address",
        "landmark": "landmark",
        "range_km": "range_km",
    }
    for raw_key, target_key in field_map.items():
        pattern = rf"\b{re.escape(raw_key)}\s*=\s*['\"]?([^,\n\"']+)"
        match = re.search(pattern, original_text, flags=re.IGNORECASE)
        if not match:
            continue
        value = match.group(1).strip()
        if target_key in host_payload:
            host_payload[target_key] = value
        if target_key in rent_payload:
            rent_payload[target_key] = value

    # Cast extracted scalar values to endpoint-compatible types.
    if host_payload["rent_per_month"] is not None:
        try:
            host_payload["rent_per_month"] = float(str(host_payload["rent_per_month"]).strip())
        except ValueError:
            host_payload["rent_per_month"] = None

    if host_payload["max_occupants"] is not None:
        try:
            host_payload["max_occupants"] = int(float(str(host_payload["max_occupants"]).strip()))
        except ValueError:
            host_payload["max_occupants"] = None

    if host_payload["pet_friendly"] is not None:
        raw_pet = str(host_payload["pet_friendly"]).strip().lower()
        if raw_pet in {"yes", "y", "true", "1"}:
            host_payload["pet_friendly"] = True
        elif raw_pet in {"no", "n", "false", "0"}:
            host_payload["pet_friendly"] = False
        else:
            host_payload["pet_friendly"] = None

    if rent_payload["range_km"] is not None:
        try:
            rent_payload["range_km"] = float(str(rent_payload["range_km"]).strip())
        except ValueError:
            rent_payload["range_km"] = 5

    if rent_payload["landmark"] is None and host_payload["address"]:
        rent_payload["landmark"] = host_payload["address"]

    return {
        "intent": intent,
        "host_payload": host_payload,
        "rent_payload": rent_payload,
    }


def determine_intent_and_payload(message: str, chat_history: list | None = None) -> dict:
    """Use LLM to select endpoint intent and extract structured payload."""
    fallback_json = '{"intent":"unknown","host_payload":{},"rent_payload":{"range_km":5}}'
    if chat_history is not None:
        formatted_messages = agent_router_stateful_prompt.format_messages(
            user_message=message,
            chat_history=chat_history,
        )
        raw = safe_llm_invoke_messages(formatted_messages, fallback_json, timeout_seconds=8)
    else:
        prompt = agent_router_prompt.format(user_message=message)
        raw = safe_llm_invoke(
            prompt,
            fallback_json,
            timeout_seconds=8,
        )
    parsed = _extract_json_object(raw)
    if not parsed:
        return _heuristic_agent_parse(message)

    intent = str(parsed.get("intent") or "unknown").strip().lower()
    if intent not in {"host_property", "rent_property", "unknown"}:
        intent = "unknown"

    host_payload = parsed.get("host_payload") if isinstance(parsed.get("host_payload"), dict) else {}
    rent_payload = parsed.get("rent_payload") if isinstance(parsed.get("rent_payload"), dict) else {}
    if "range_km" not in rent_payload or rent_payload.get("range_km") is None:
        rent_payload["range_km"] = 5

    result = {
        "intent": intent,
        "host_payload": host_payload,
        "rent_payload": rent_payload,
    }

    # If model classifies as unknown, apply lightweight lexical fallback for intent only.
    # This keeps payload extraction from LLM while improving routing reliability.
    if result["intent"] == "unknown":
        heuristic = _heuristic_agent_parse(message)
        if heuristic.get("intent") in {"host_property", "rent_property"}:
            result["intent"] = heuristic["intent"]

    # Backfill missing fields from heuristic extraction when LLM omits structured payload values.
    heuristic = _heuristic_agent_parse(message)
    if result["intent"] == "host_property":
        for key, value in (heuristic.get("host_payload") or {}).items():
            if result["host_payload"].get(key) in (None, "") and value not in (None, ""):
                result["host_payload"][key] = value
    if result["intent"] == "rent_property":
        for key, value in (heuristic.get("rent_payload") or {}).items():
            if result["rent_payload"].get(key) in (None, "") and value not in (None, ""):
                result["rent_payload"][key] = value

    return result

# Define tool-call wrappers for external services.
@tool
def nominatim_search_tool(query: str) -> list[dict]:
    """Fetch raw Nominatim search payload for a text query."""
    headers = {
        # Nominatim usage policy requires identifying User-Agent.
        "User-Agent": "real-estate-assistant/1.0 (contact: local-dev)",
    }
    params = {
        "q": query,
        "format": "json",
        "addressdetails": 1,
        "limit": 1,
    }
    response = requests.get(NOMINATIM_URL, params=params, headers=headers, timeout=15)
    response.raise_for_status()
    payload = response.json()
    return payload if isinstance(payload, list) else []


@tool
def overpass_search_tool(latitude: float, longitude: float, range_km: float) -> list[dict]:
    """Fetch raw Overpass elements for configured amenity categories."""
    overpass_query = (
        f"[out:json];"
        "("
        f"node[shop=mall](around:{range_km * 1000},{latitude},{longitude});"
        f"node[amenity=school](around:{range_km * 1000},{latitude},{longitude});"
        f"node[railway=station](around:{range_km * 1000},{latitude},{longitude});"
        f"node[highway=bus_stop](around:{range_km * 1000},{latitude},{longitude});"
        ");"
        f"out body {OVERPASS_NODE_LIMIT};"
    )
    headers = {"User-Agent": "real-estate-assistant/1.0 (contact: local-dev)"}
    last_error: Exception | None = None
    for base_url in OVERPASS_URLS:
        try:
            response = requests.get(base_url, params={"data": overpass_query}, headers=headers, timeout=12)
            # Overpass frequently responds with transient 5xx under load; try next mirror.
            if response.status_code >= 500:
                continue

            response.raise_for_status()
            payload = response.json()
            elements = payload.get("elements", []) if isinstance(payload, dict) else []
            return elements if isinstance(elements, list) else []
        except (requests.RequestException, ValueError) as exc:
            last_error = exc
            continue

    if last_error is not None:
        raise last_error
    raise requests.RequestException("All Overpass endpoints failed.")


def fetch_coordinates_with_nominatim(address: str) -> dict:
    """Fetch coordinates for a given address using Nominatim API."""
    cache_key = address.strip().lower()
    cached = _geocode_cache.get(cache_key)
    if cached and (time.time() - cached[0]) < CACHE_TTL_SECONDS:
        return cached[1]

    # Try a few query variants before declaring the address invalid.
    query_variants = [
        address,
        address.replace(" RAILWAY STATION", ""),
        f"{address}, India",
    ]

    service_unavailable = False
    for query in query_variants:
        try:
            payload = nominatim_search_tool.invoke({"query": query})
            if payload:
                location = payload[0]
                result = {
                    "latitude": float(location["lat"]),
                    "longitude": float(location["lon"]),
                }
                _geocode_cache[cache_key] = (time.time(), result)
                return result
        except requests.RequestException:
            service_unavailable = True
            continue

    if service_unavailable:
        raise HTTPException(status_code=502, detail="Geocoding service is unavailable.")

    raise HTTPException(status_code=400, detail="Unable to fetch coordinates for the address.")


def fetch_amenities_with_overpass(latitude: float, longitude: float, range_km: float) -> List[dict]:
    """Fetch nearby amenities from Overpass API."""
    cache_key = f"overpass:{round(latitude, 4)}:{round(longitude, 4)}:{round(range_km, 1)}"
    cached = _amenities_cache.get(cache_key)
    if cached and (time.time() - cached[0]) < CACHE_TTL_SECONDS:
        return cached[1]

    elements = overpass_search_tool.invoke(
        {"latitude": latitude, "longitude": longitude, "range_km": range_km}
    )

    result: List[dict] = []
    for item in elements:
        tags = item.get("tags", {})
        title = tags.get("name") or tags.get("brand") or tags.get("operator")

        # Skip records that cannot produce a meaningful amenity entry.
        if not title:
            continue

        if tags.get("shop") == "mall":
            category = "mall"
        elif tags.get("amenity") == "school":
            category = "school"
        elif tags.get("railway") == "station":
            category = "railway_station"
        elif tags.get("highway") == "bus_stop":
            category = "bus_station"
        else:
            category = "other"

        address_parts = [
            tags.get("addr:housenumber"),
            tags.get("addr:street"),
            tags.get("addr:city"),
        ]
        address = ", ".join(part for part in address_parts if part) or tags.get("addr:full")

        result.append(
            {
                "source": "overpass",
                "category": category,
                "title": title,
                "rating": None,
                "reviews": None,
                "address": address,
                "phone": tags.get("phone") or tags.get("contact:phone"),
                "website": tags.get("website") or tags.get("contact:website"),
                "direction": None,
                "latitude": item.get("lat"),
                "longitude": item.get("lon"),
            }
        )
        if len(result) >= MAX_AMENITIES_RESULTS:
            break

    _amenities_cache[cache_key] = (time.time(), result)
    return result


def fetch_amenities_with_searchapi(landmark: str) -> List[dict]:
    """Fetch nearby amenities from SearchApi Google Local."""
    cache_key = f"searchapi:{landmark.strip().lower()}"
    cached = _amenities_cache.get(cache_key)
    if cached and (time.time() - cached[0]) < CACHE_TTL_SECONDS:
        return cached[1]

    api_key = os.getenv("SEARCHAPI_API_KEY")
    if not api_key:
        raise ValueError("SEARCHAPI_API_KEY is not configured.")

    category_queries = {
        "mall": "famous malls",
        "school": "top schools",
        "railway_station": "railway station",
        "bus_station": "bus stand",
    }

    amenities: List[dict] = []
    for category, query in category_queries.items():
        params = {
            "engine": "google_local",
            "q": f"{query} near {landmark}",
            "location": landmark,
            "api_key": api_key,
            "hl": "en",
            "gl": "in",
        }
        response = requests.get(SEARCHAPI_URL, params=params, timeout=20)
        response.raise_for_status()
        payload = response.json()

        for item in payload.get("local_results", []):
            title = item.get("title")
            if not title:
                continue

            amenities.append(
                {
                    "source": "searchapi",
                    "category": category,
                    "title": title,
                    "rating": item.get("rating"),
                    "reviews": item.get("reviews"),
                    "address": (item.get("extensions") or [None])[0],
                    "phone": item.get("phone"),
                    "website": item.get("website"),
                    "direction": item.get("direction"),
                    "latitude": (item.get("gps_coordinates") or {}).get("latitude"),
                    "longitude": (item.get("gps_coordinates") or {}).get("longitude"),
                }
            )
            if len(amenities) >= MAX_AMENITIES_RESULTS:
                _amenities_cache[cache_key] = (time.time(), amenities)
                return amenities

    _amenities_cache[cache_key] = (time.time(), amenities)
    return amenities


def normalize_owner_suggestion(owner_name: str, suggestion: str) -> str:
    """Guarantee a positive owner-facing confirmation message."""
    fallback = (
        f"Thanks for listing the property {owner_name}, We will show to the customers "
        "and they will contact you shortly if they were interested."
    )
    if not suggestion:
        return fallback

    lower_text = suggestion.lower()
    blocked_phrases = [
        "temporarily unavailable",
        "unable",
        "error",
        "failed",
    ]
    if any(phrase in lower_text for phrase in blocked_phrases):
        return fallback

    return suggestion.strip()


def haversine_distance_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Calculate great-circle distance between two points in kilometers."""
    r = 6371.0
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = (
        math.sin(dlat / 2) ** 2
        + math.cos(math.radians(lat1))
        * math.cos(math.radians(lat2))
        * math.sin(dlon / 2) ** 2
    )
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return r * c


def fetch_travel_distance_km(origin_lat: float, origin_lon: float, dest_lat: float, dest_lon: float) -> float | None:
    """Return road distance in km using OSRM; fallback caller handles None."""
    route_url = f"{OSRM_ROUTE_URL}/{origin_lon},{origin_lat};{dest_lon},{dest_lat}"
    try:
        response = requests.get(route_url, params={"overview": "false"}, timeout=8)
        response.raise_for_status()
        payload = response.json()
        routes = payload.get("routes", [])
        if not routes:
            return None
        # OSRM distance is in meters.
        return float(routes[0].get("distance", 0.0)) / 1000.0
    except requests.RequestException:
        return None


def build_nearby_landmarks(amenities: List[dict], max_items: int = 10) -> List[str]:
    """Extract unique landmark names from amenities payload."""
    landmarks: List[str] = []
    seen: set[str] = set()

    for item in amenities:
        title = item.get("title")
        if isinstance(title, str) and title.strip():
            cleaned = title.strip()
            key = cleaned.lower()
            if key not in seen:
                seen.add(key)
                landmarks.append(cleaned)
                if len(landmarks) >= max_items:
                    break

    return landmarks


def amenity_with_property_distance(property_lat: float, property_lon: float, amenity: dict) -> dict:
    """Return amenity dict enriched with distance from current property when coordinates exist."""
    enriched = dict(amenity)
    amenity_lat = amenity.get("latitude")
    amenity_lon = amenity.get("longitude")
    if isinstance(amenity_lat, (int, float)) and isinstance(amenity_lon, (int, float)):
        enriched["distance_from_property_km"] = round(
            haversine_distance_km(property_lat, property_lon, float(amenity_lat), float(amenity_lon)),
            2,
        )
    else:
        enriched["distance_from_property_km"] = None
    return enriched


def is_famous_landmark(amenity: dict) -> bool:
    """Heuristic check for landmark prominence."""
    title = str(amenity.get("title") or "").strip().lower()
    if not title:
        return False

    category = amenity.get("category")
    rating = amenity.get("rating")
    reviews = amenity.get("reviews")

    score = 0
    if category in {"mall", "railway_station", "bus_station"}:
        score += 1
    if isinstance(rating, (int, float)) and float(rating) >= 4.2:
        score += 1
    if isinstance(reviews, (int, float)) and float(reviews) >= 200:
        score += 1

    famous_keywords = [
        "junction",
        "central",
        "main",
        "mall",
        "beach",
        "fort",
        "temple",
        "museum",
        "park",
        "stadium",
    ]
    if any(keyword in title for keyword in famous_keywords):
        score += 1

    return score >= 2


def build_famous_landmarks_for_property(
    property_lat: float,
    property_lon: float,
    amenities: List[dict],
    max_items: int = MAX_LANDMARK_RESULTS,
) -> List[dict]:
    """Build per-property famous landmarks with distance."""
    candidates: List[dict] = []
    seen: set[str] = set()

    for amenity in amenities:
        if not is_famous_landmark(amenity):
            continue

        enriched = amenity_with_property_distance(property_lat, property_lon, amenity)
        distance_km = enriched.get("distance_from_property_km")
        title = enriched.get("title")
        if not isinstance(title, str) or not title.strip() or distance_km is None:
            continue

        key = title.strip().lower()
        if key in seen:
            continue
        seen.add(key)

        candidates.append(
            {
                "title": title.strip(),
                "category": enriched.get("category"),
                "distance_km": float(distance_km),
            }
        )

    candidates.sort(key=lambda item: item["distance_km"])
    return candidates[:max_items]


def sanitize_amenities(amenities: List[dict]) -> List[dict]:
    """Keep only amenity records that can be represented meaningfully in API output."""
    keys = [
        "source",
        "category",
        "title",
        "rating",
        "reviews",
        "address",
        "phone",
        "website",
        "direction",
        "latitude",
        "longitude",
        "distance_from_property_km",
    ]
    cleaned: List[dict] = []

    for item in amenities:
        if not isinstance(item, dict):
            continue

        normalized = {key: item.get(key) for key in keys}
        if not normalized["title"]:
            continue

        if all(normalized[key] is None for key in keys):
            continue

        cleaned.append(normalized)
        if len(cleaned) >= MAX_AMENITIES_RESULTS:
            break

    return cleaned

@app.post("/host-property")
async def host_property(request: HostPropertyRequest):
    """
    Endpoint to host a property for rent.
    """
    # Fetch latitude and longitude using Nominatim API
    coordinates = fetch_coordinates_with_nominatim(request.address)

    # Store property details in the database
    db = SessionLocal()
    try:
        new_property = Property(
            owner_name=request.owner_name,
            rent_per_month=request.rent_per_month,
            max_occupants=request.max_occupants,
            property_type=request.property_type,
            floor=request.floor,
            pet_friendly=request.pet_friendly,
            latitude=coordinates["latitude"],
            longitude=coordinates["longitude"]
        )
        db.add(new_property)
        db.commit()
        db.refresh(new_property)
        property_id = new_property.id
    except SQLAlchemyError as exc:
        db.rollback()
        raise HTTPException(status_code=500, detail=f"Failed to save property details: {exc}")
    finally:
        db.close()

    # Use LLM to suggest improvements for the property listing
    property_details = f"Owner: {request.owner_name}, Rent: {request.rent_per_month}, Max Occupants: {request.max_occupants}, Type: {request.property_type}, Floor: {request.floor}, Pet Friendly: {request.pet_friendly}, Address: {request.address}"
    prompt = host_prompt.format(owner_name=request.owner_name, property_details=property_details)
    raw_suggestion = safe_llm_invoke(
        prompt,
        f"Thanks for listing the property {request.owner_name}, We will show to the customers and they will contact you shortly if they were interested.",
    )
    suggestions_text = normalize_owner_suggestion(request.owner_name, raw_suggestion)

    return {
        "message": "Property hosted successfully.",
        "property_id": property_id,
        "suggestions": suggestions_text
    }

@app.post("/rent-property", response_model=RentPropertyResponse)
async def rent_property(request: RentPropertyRequest):
    """
    Endpoint to search for properties and nearby amenities.
    """
    # Fetch latitude and longitude for the landmark
    coordinates = fetch_coordinates_with_nominatim(request.landmark)
    latitude = coordinates["latitude"]
    longitude = coordinates["longitude"]

    # Fetch properties from the database and filter by distance range.
    db = SessionLocal()
    properties = db.query(Property).all()
    db.close()

    filtered_properties = []
    travel_distance_cache: dict[tuple[float, float], float | None] = {}
    candidate_properties: list[tuple[Property, float]] = []
    for prop in properties:
        if prop.latitude is None or prop.longitude is None:
            continue

        # Quick pre-filter by straight-line distance before slower routing call.
        straight_line_km = haversine_distance_km(latitude, longitude, prop.latitude, prop.longitude)
        if straight_line_km > (request.range_km * 1.5):
            continue

        candidate_properties.append((prop, straight_line_km))

    # Route-distance checks are expensive; only process nearest candidates.
    candidate_properties = sorted(candidate_properties, key=lambda item: item[1])[:MAX_PROPERTIES_FOR_ROUTING]

    for prop, straight_line_km in candidate_properties:
        cache_key = (prop.latitude, prop.longitude)
        if cache_key not in travel_distance_cache:
            travel_distance_cache[cache_key] = fetch_travel_distance_km(
                latitude,
                longitude,
                prop.latitude,
                prop.longitude,
            )

        travel_km = travel_distance_cache[cache_key]
        effective_distance_km = travel_km if travel_km is not None else straight_line_km

        if effective_distance_km <= request.range_km:
            filtered_properties.append((prop, effective_distance_km))

    # Fetch nearby amenities using SearchApi first, then fallback to Overpass.
    amenities: List[dict] = []
    amenities_fetch_failed = False
    try:
        amenities = fetch_amenities_with_searchapi(request.landmark)
    except Exception:
        try:
            amenities = fetch_amenities_with_overpass(latitude, longitude, request.range_km)
        except Exception:
            amenities_fetch_failed = True

    amenities = sanitize_amenities(amenities)

    # Use LLM to generate a summary of the search results
    search_results = []
    for prop, distance_km in sorted(filtered_properties, key=lambda item: item[1]):
        property_amenities = [
            amenity_with_property_distance(prop.latitude, prop.longitude, amenity)
            for amenity in amenities
        ]
        property_landmarks = build_famous_landmarks_for_property(
            prop.latitude,
            prop.longitude,
            property_amenities,
        )

        search_results.append(
            {
                "id": prop.id,
                "owner_name": prop.owner_name,
                "rent_per_month": prop.rent_per_month,
                "latitude": prop.latitude,
                "longitude": prop.longitude,
                "distance_km": round(distance_km, 2),
                "amenities": property_amenities,
                "nearby_landmarks": property_landmarks,
            }
        )
    prompt = rent_prompt.format(search_results=search_results[:10], amenities=amenities[:20])
    fallback_summary = (
        "Found matching properties, but nearby amenities are temporarily unavailable."
        if amenities_fetch_failed
        else "Found properties and amenities, but summary generation is temporarily unavailable."
    )
    summary_text = safe_llm_invoke(prompt, fallback_summary, timeout_seconds=6)
    aggregated_landmarks: List[dict] = []
    seen_landmarks: set[str] = set()
    for item in search_results:
        for landmark in item.get("nearby_landmarks", []):
            key = landmark["title"].lower()
            if key in seen_landmarks:
                continue
            seen_landmarks.add(key)
            aggregated_landmarks.append(landmark)

    aggregated_landmarks.sort(key=lambda item: item["distance_km"])
    nearby_landmarks = aggregated_landmarks[:MAX_LANDMARK_RESULTS]

    return {
        "properties": search_results,
        "amenities": amenities,
        "nearby_landmarks": nearby_landmarks,
        "summary": summary_text
    }


@app.post("/agent-chat", response_model=AgentChatResponse)
async def agent_chat(request: AgentChatRequest):
    """Agentic entrypoint that auto-routes user chat to host/rent workflows."""
    if _is_memory_question(request.message):
        memory_answer = _build_memory_answer(request.session_id)
        _append_session_history(
            request.session_id,
            request.message,
            json.dumps({
                "intent": "memory_lookup",
                "selected_endpoint": "/agent-chat",
                "answer": memory_answer,
            }),
        )
        return {
            "intent": "memory_lookup",
            "selected_endpoint": "/agent-chat",
            "extracted_payload": {"session_id": request.session_id},
            "result": {"message": memory_answer},
        }

    history = _get_session_history(request.session_id)
    decision = determine_intent_and_payload(request.message, chat_history=history)
    intent = decision.get("intent", "unknown")

    if intent == "host_property":
        payload = decision.get("host_payload") or {}
        required = [
            "owner_name",
            "rent_per_month",
            "max_occupants",
            "property_type",
            "floor",
            "pet_friendly",
            "address",
        ]
        missing = [key for key in required if payload.get(key) in (None, "")]
        if missing:
            raise HTTPException(
                status_code=422,
                detail=f"Need more details for host-property: {', '.join(missing)}",
            )

        host_request = HostPropertyRequest(**payload)
        result = await host_property(host_request)
        _append_session_history(
            request.session_id,
            request.message,
            json.dumps({
                "intent": intent,
                "selected_endpoint": "/host-property",
                "extracted_payload": host_request.model_dump(),
            }),
        )
        return {
            "intent": intent,
            "selected_endpoint": "/host-property",
            "extracted_payload": host_request.model_dump(),
            "result": result,
        }

    if intent == "rent_property":
        payload = decision.get("rent_payload") or {}
        if not payload.get("landmark"):
            raise HTTPException(status_code=422, detail="Need landmark for rent-property search.")

        rent_request = RentPropertyRequest(**payload)
        result = await rent_property(rent_request)
        _append_session_history(
            request.session_id,
            request.message,
            json.dumps({
                "intent": intent,
                "selected_endpoint": "/rent-property",
                "extracted_payload": rent_request.model_dump(),
            }),
        )
        return {
            "intent": intent,
            "selected_endpoint": "/rent-property",
            "extracted_payload": rent_request.model_dump(),
            "result": result,
        }

    raise HTTPException(
        status_code=422,
        detail="Could not determine intent. Please mention whether you want to host or rent a property.",
    )


@app.post("/agent-chat/reset-memory")
async def reset_agent_chat_memory(session_id: str | None = None):
    """Reset short-term memory for one session, or all sessions when omitted."""
    if session_id is None or not session_id.strip():
        cleared_count = len(_agent_conversation_history)
        _agent_conversation_history.clear()
        return {
            "message": "Agent chat memory reset for all sessions.",
            "cleared_sessions": cleared_count,
        }

    key = session_id.strip()
    if key in _agent_conversation_history:
        del _agent_conversation_history[key]
        return {
            "message": f"Agent chat memory reset for session '{key}'.",
            "session_id": key,
            "cleared": True,
        }

    return {
        "message": f"No memory found for session '{key}'.",
        "session_id": key,
        "cleared": False,
    }