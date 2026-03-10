# Course: Build and Deploy Intelligent Conversational Agents

## Capstone Project Overview

This capstone project gives you a hands-on opportunity to apply everything learned across the course. By selecting one of the provided project scenarios, you will design and build an **Intelligent Conversational Agent** using modern AI engineering practices. Your goal is to architect, develop, test, and present a fully functional **LangChain-powered backend application** with LLM integration, modular code structure, observability, memory, prompt management, and tool calling.

This project simulates real-world development of an AI agent—complete with authentication, context awareness, multi-step workflows, and high-quality API responses.

---

### Your Task
[Priority High]
1. Select one of the provided **project scenarios** (e.g., Real Estate Assistant, News Summary Agent, Finance Advisor, etc.).
2. Design the **application architecture** (modules, services, routes, data models, and pipelines).
3. Build a **LangChain-powered backend application** that implements:
   - Core agent endpoints (query, recommendation, context-handling)
   - Structured validation with **Pydantic models**
   - **LLM integration with LangChain** 
   - **Tool Calling** and external API integrations 
   - **Multi-step workflows using LCEL** 
   - **Short-term memory** for multi-turn conversations 
4. Add **Langfuse observability** for tracing agent workflows .
5. Implement clean **error handling**, modular architecture, and async logic.
**[Priority Low]**

[Priority High]
Prepare a **final project presentation and demo**.

---
## Third Party API:
  - **Disclaimer**: If the third-party API link in the project specification is not working or has become a paid service, please explore and use another third-party API.

  - You are free to use any third-party API; the ones provided are only for reference.
    

## Project Objectives

By completing this project, you will:

- Design and build a modular, scalable **AI conversational backend**.
- Apply **LangChain** techniques for prompt management, LLM calls, and multi-step logic.
- Use **tool calling** to integrate external APIs.
- Implement **short-term session memory** for multi-turn conversations.

---

## Project Workflow and Milestones

### 1. Requirement Analysis & Planning
- Select your scenario.
- Define user flows, modules, tools, and endpoints.
- Submit a one-page architecture plan.

### 2. Environment Setup
- Create FastAPI project structure.  
- Configure env variables for API keys and Auth0.
- Install core dependencies:  
  `fastapi`, `uvicorn`, `pydantic`, `langchain`, `langfuse`, `python-dotenv`, `requests`.

### 3. Core API Development
- Implement agent endpoints.
- Use Pydantic models for robust validation.
- Organize logic into services.

### 4. LLM Integration & Prompt Management
- Integrate Gemini via LangChain.
- Implement prompts and prompt templates.
- Ensure consistent agent responses.

### 5. Tool Calling & External API Integration 
- Enable agent tool calling.
- Integrate external APIs from your selected scenario.
- Ensure error-resilient API communication.

### 6. Multi-Step LLM Chains with LCEL
- Build workflows such as:
  - Query → Fetch data → Summarize → Recommend
  - Context-aware decision routing

### 7. Agent Memory 
- Add short-term session memory.
- Store user preferences, last queries, and preferences during session.

### 8. Observability 
- Instrument all agent workflows using Langfuse.
- Capture traces, tool calls, inputs, outputs, and errors.

### 9. Documentation & Presentation
- Create README, architecture diagrams, and workflow explanation.
- Present a 3–4 minute live demo.




---

## Deliverables & Sprint Timeline

| Sprint # | Deliverable | Description |
|---------|-------------|-------------|
| Sprint 11 | **Project Approach Document** | Problem, objectives, modules, tools, and architecture plan |
| Sprint 11 | **Backend Setup** | Project structure, dependencies, and env configuration |
| Sprint 12 | **Core API Development** | Core routes, validation, responses |
| Sprint 12 | **LLM Integration** | LLM calls, prompt templates |
| Sprint 12 | **Tool Calling + Memory** | Multi-step logic, external API integration |
| Sprint 14/15 | **Final Demo** | Presentation + documentation |

---

## Guidelines for Success

- Start early and build incrementally.
- Prioritize correctness and clarity over complexity.
- Write clean, modular code—avoid large monolithic files.
- Test endpoints frequently.
- Keep API responses consistent and predictable.
- Use Swagger UI to verify models & schema.
- Maintain clear and structured commit history.

---

## Evaluation Criteria

| Parameter | Weight |
|----------|--------|
| Architecture & System Design | 13% |
| API Development & Validation | 28% |
| LLM Integration | 18% |
| Multi-step Logic & Context Awareness | 18% |
| Documentation & Presentation | 10% |
| Creativity & Enhancements | 13% |

Total: **100%**

---

## Project Scenario Options

Choose from any of the rewritten scenarios:

- Real Estate Assistant  
- World News Summary Agent  
- Personalized Nutrition Planner  
- Finance & Budget Advisor  
- Personal Wellbeing Coach  
- Or any other provided scenario  

Each scenario includes:
- Problem statement  
- Functional requirements  
- API tools  
- Technical details  

---

## Conclusion

This capstone project helps you apply your knowledge to build a **real, deployable conversational agent** with all essential AI engineering components:

- LLMs  
- Prompts  
- Tools  
- Memory  
- Frontend  
- Auth  
- Observability  
- Testing  

By completing this project, you’ll have a **portfolio-ready AI agent**, demonstrating your ability to design and implement **intelligent, reliable conversational systems**—end to end. 
