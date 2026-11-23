# GeoAI Agent — Q&A platform (Streamlit version, 2026-11-23)
# The agent from geoai-agent-skeleton-gemini.py is the backend; this file is
# only the chat UI shell around it. Run locally or deploy free.
#
# Setup:
#   pip install streamlit
#   Set the key WITHOUT putting it in this file:
#     - local: create .streamlit/secrets.toml with GEMINI_API_KEY = "AQ...."
#     - Streamlit Community Cloud / HF Spaces: paste it in the app's Secrets settings
# Run:
#   streamlit run geoai-agent-app.py

import uuid

import requests
import streamlit as st
from geopy.geocoders import Nominatim
from langchain.agents import create_agent
from langchain_core.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.checkpoint.memory import MemorySaver
from pydantic import BaseModel, Field

# --- Tools: same two as the skeleton ---
class GeocodeInput(BaseModel):
    place: str = Field(description="Place name or address in New York City, e.g. 'Times Square'")

@tool(args_schema=GeocodeInput)
def geocode(place: str) -> str:
    """Geocode a place name in New York City to latitude/longitude."""
    loc = Nominatim(user_agent="geoai-agent-demo", timeout=10).geocode(f"{place}, New York City")
    if loc is None:
        return "NOT_FOUND: no match for this place name"
    return f"{loc.address}: lat={loc.latitude:.5f}, lon={loc.longitude:.5f}"

class RouteInput(BaseModel):
    origin_lat: float = Field(ge=40.4, le=40.95, description="Origin latitude, must be inside NYC")
    origin_lon: float = Field(ge=-74.3, le=-73.7, description="Origin longitude, must be inside NYC")
    dest_lat: float = Field(ge=40.4, le=40.95, description="Destination latitude, must be inside NYC")
    dest_lon: float = Field(ge=-74.3, le=-73.7, description="Destination longitude, must be inside NYC")

@tool(args_schema=RouteInput)
def route(origin_lat: float, origin_lon: float, dest_lat: float, dest_lon: float) -> str:
    """Driving distance and duration between two NYC coordinates."""
    url = f"https://router.project-osrm.org/route/v1/driving/{origin_lon},{origin_lat};{dest_lon},{dest_lat}"
    r = requests.get(url, params={"overview": "false"}, timeout=20)
    data = r.json()
    if data.get("code") != "Ok" or not data.get("routes"):
        return "NOT_FOUND: no route between these points"
    leg = data["routes"][0]
    return f"distance={leg['distance']/1000:.1f} km, duration={leg['duration']/60:.1f} min"

# --- Agent (same as skeleton). interrupt_before is kept, but this demo
# auto-approves each tool call in the loop below, so the web user is never
# blocked. To require approval in the UI, replace the loop with Approve /
# Reject buttons that call agent.invoke(None, config) or reject the call.
@st.cache_resource
def build_agent():
    model = ChatGoogleGenerativeAI(
        model="gemini-3.5-flash-lite",
        google_api_key=st.secrets["GEMINI_API_KEY"],
        temperature=0,
    )
    return create_agent(
        model, [geocode, route],
        checkpointer=MemorySaver(),
        interrupt_before=["tools"],
    )

agent = build_agent()

st.title("GeoAI Agent — NYC Q&A")
st.caption("Ask about driving in New York City. The agent geocodes places and computes routes itself.")

if "messages" not in st.session_state:
    st.session_state.messages = []
if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())

for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

if q := st.chat_input("e.g. How long does it take to drive from JFK Airport to Times Square?"):
    st.session_state.messages.append({"role": "user", "content": q})
    with st.chat_message("user"):
        st.markdown(q)
    with st.chat_message("assistant"):
        config = {"configurable": {"thread_id": st.session_state.thread_id}}
        result = agent.invoke({"messages": [("user", q)]}, config)
        while agent.get_state(config).next:
            result = agent.invoke(None, config)  # auto-approve tool calls
        answer = result["messages"][-1].content
        st.markdown(answer)
    st.session_state.messages.append({"role": "assistant", "content": answer})
