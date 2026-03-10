  # Real Estate Assistant & Home Recommendation Agent – Your AI-Powered Home Search Companion

An intelligent conversational agent that helps users discover suitable homes based on budget, location, commute preferences, lifestyle needs, and neighborhood insights—powered by real-time geocoding, place information, and contextual reasoning.

---

## Problem Statement: Real Estate Assistant & Home Recommendation Agent

### Context

Searching for a new home is often stressful and time-consuming. Property seekers must navigate hundreds of listings across multiple platforms, evaluate neighborhoods without reliable local knowledge, compare amenities, and determine how well each property matches their lifestyle. This scattered process leads to confusion, frustration, and poor decision-making.

**Common Challenges:**

- **Too Many Listings:** Users struggle to filter irrelevant or low-quality listings across different real estate portals.  
- **Lack of Neighborhood Knowledge:** People often lack clarity about schools, hospitals, commute times, or nearby amenities.  
- **Difficult Comparisons:** It's hard to compare properties across price, size, location, and lifestyle suitability.  
- **Limited Personalization:** Most real-estate platforms do not provide customized recommendations based on lifestyle, commute needs, or family size.  
- **Fragmented Tools:** Users rely on different apps for maps, commute planning, and neighborhood data.  
- **No Session Awareness:** Real estate apps rarely remember user preferences, budgets, or previously liked/rejected homes during a session.

### Project Goal

Develop an **AI-powered Real Estate Assistant** that provides a conversational interface to help users find suitable homes, compare neighborhoods, evaluate commute preferences, and receive personalized lifestyle-matched suggestions—all while maintaining context during the session.

---

## Problem Description

This project involves building a **full-stack Real Estate Assistant Application** that:

- Conversationally interacts with users to understand their budget, location preferences, family size, commute needs, and must-have amenities.
- Fetches geographic and neighborhood details using **external APIs (Note: The APIs mentioned below are suggestions. You are free to use other geocoding/travel APIs that suit your needs):**
  - **Nominatim (OpenStreetMap Geocoding API — Free)** — for address/geolocation lookup, coordinates, locality identification, and region suggestions.  
    API URL: https://nominatim.openstreetmap.org/
  - **OpenTripMap API** — for retrieving places of interest such as schools, gyms, hospitals, transportation hubs, and lifestyle amenities near a property.  
    API URL: https://opentripmap.io/

**Disclaimer: The API key generation link provided may be subject to a paid version. You have full discretion to switch to an alternative API key or plan based on your project requirements.**

- Feel free to test with other Geo coding and travel APIs.
- Recommends home listings that match user preferences.
- Presents neighborhood insights including commute estimates, essential services, and lifestyle fit.
- Maintains session memory: preferred locations, budget, liked/disliked homes.
- Provides observability using Langfuse for agent reasoning, multi-step workflows, and API/tool calls.

---

## Functional Requirements

### 1. Conversational Property Assistant  
A chat-based AI agent capable of:
- Understanding user requirements:
  - Budget  
  - Preferred city/area  
  - Family size  
  - Commute time or mode  
  - Amenities  
- Maintaining context across the conversation.
- Automatically deciding when to:
  - Look up coordinates using Nominatim  
  - Fetch nearby places using OpenTripMap  
  - Recommend properties  
  - Offer lifestyle-based suggestions  

### 2. External API Integrations

You can use the suggested APIs below or choose alternative geocoding/travel APIs that better fit your project needs.

#### **A. Nominatim (OpenStreetMap Geocoding API – Free)**
Used for:
- Converting user-entered locations to latitude/longitude  
- Reverse-geocoding property coordinates  
- Identifying neighborhood names  
- Suggesting nearby localities within the budget  

#### **B. OpenTripMap API **
Used for retrieving:
- Nearby schools  
- Hospitals & clinics  
- Public transport hubs  
- Gyms, parks, malls  
- Distance & walking time to amenities  

The agent must automatically route the correct API based on intent.

### 3. Home Recommendations

- Provide recommended properties based on budget, location, and needs.
- Include:
  - Price  
  - Size (sqft/BHK)  
  - Locality  
  - Pros & cons  

### 4. Neighborhood & Commute Insights

- Analyze nearby amenities using OpenTripMap.
- Provide:
  - Commute estimates  
  - Family-friendliness  
  - Connectivity  
  - Entertainment & utilities  

### 5. Lifestyle-Matched Suggestions

- Family-friendly areas  
- Professional-friendly areas  
- Budget-friendly zones  
- Luxury zones  

### 6. Multi-Step Workflow Implementation

Examples:
- Budget → Geolocation → Neighborhoods → Homes  
- Property → Coordinates → Amenities → Insights  
- Lifestyle needs → Area suggestions  

### 7. Short-Term Session Memory

Remembers:
- Budget  
- Preferred localities  
- Commute needs  
- Liked/rejected homes  

 ### 8. FastAPI Backend

- Secure endpoints (if authentication is implemented)  
- LangChain agent workflow  
- Tool integrations  
- Session memory  
- Langfuse logging  

### 9. Observability with Langfuse

Tracks:
- LLM outputs  
- API calls  
- Workflow logic  
- Errors & latency  

### 10. Secure Authentication with Auth0 

**Note:** Authentication is optional. If implemented:
- Users must log in before accessing the features.
- The frontend must obtain a JWT and pass it to the FastAPI backend.(Optional)
- The FastAPI backend must validate the JWT before processing agent requests.(Mandatory) 

### 11. Frontend (Streamlit/React/HTML,CSS,JavaScript)(Optional)

**Note:** Frontend design is optional. If implemented, consider including:
- Login/logout with Auth0 (if authentication is implemented)  
- Chat interface  
- Property cards  
- Maps and amenities list  
- Buttons for commute, comparison, and insights 

---

## Technical Details

**Languages:** Python (Backend), TypeScript/JavaScript (Frontend)

### Libraries & Tools

| Tool | Purpose |
|------|---------|
| `fastapi` | Build backend API |
| `uvicorn` | Run FastAPI server |
| `langchain` | Build LLM workflows and agent pipelines |
| `langfuse` | Observability and tracing for LLM applications |
| `requests` | Communicate with external APIs (SerpAPI, Rise API) |
| `python-jose` | JWT token validation |
| `python-dotenv` | Manage environment variables securely |
| `react` | Build frontend user interface |
| `vite` | The Build Tool for the Web |
| `Streamlit` | Faster way to build and share Data Apps |
| `auth0-react` | Auth0 authentication integration for React |

### Environment Variables
Add all the necessary environment variables. In the project, you can use either the Gemini model or the Azure OpenAI model for LLM calls.

| Variable | Purpose |
|---------|---------|
| `GEMINI_API_KEY` | Gemini API key for LLM integrations |
| `GEMINI_MODEL_NAME` | Gemini Model name for LLM integrations |
| `GEMINI_BASE_URL` | Gemini Base url for LLM integrations |
| `AUTH0_DOMAIN` | Auth0 domain for authentication |
| `AUTH0_CLIENT_ID` | Auth0 client ID |
| `AUTH0_CLIENT_SECRET` | Auth0 client secret |
| `AUTH0_AUDIENCE` | Auth0 API audience identifier |
| `LANGFUSE_SECRET_KEY` | Langfuse secret key for authentication |
| `LANGFUSE_PUBLIC_KEY` | Langfuse public key for authentication |
| `LANGFUSE_HOST` | Langfuse base url for authentication |

---

## Final Deliverables

1. FastAPI Backend with LangChain Agent (Required)
2. External API Integration (Required - can use Nominatim/OpenTripMap or alternative geocoding/travel APIs)
3. Langfuse Observability (Required)
4. Frontend Application (React/Streamlit/HTML,CSS,JavaScript or any UI implementation in the project)
5. Auth0 Authentication (Backend required)
6. README with diagrams and workflows  

---

## Goal

Build a production-grade intelligent real estate assistant with LLM reasoning, API integrations, geospatial analysis, workflow orchestration, authentication, and a modern UI.

