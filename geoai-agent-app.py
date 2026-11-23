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

import folium
import requests
import streamlit as st
from geopy.geocoders import Nominatim
from langchain.agents import create_agent
from langchain_core.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.checkpoint.memory import MemorySaver
from pydantic import BaseModel, Field
from streamlit_folium import st_folium

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
# The key is read from Streamlit Secrets first, then from an environment
# variable (how AWS App Runner / ECS pass secrets to the container).
import os

def _get_api_key():
    try:
        return st.secrets["GEMINI_API_KEY"]
    except Exception:
        return os.environ.get("GEMINI_API_KEY")

def _message_text(message) -> str:
    """Extract plain text from a message. Newer langchain versions return
    content as a list of blocks (e.g. [{'type': 'text', 'text': '...'}]),
    not a plain string, so printing it raw shows the whole structure."""
    content = message.content
    if isinstance(content, str):
        return content
    parts = []
    for block in content:
        if isinstance(block, dict) and block.get("type") == "text":
            parts.append(block.get("text", ""))
        elif hasattr(block, "text"):
            parts.append(block.text)
    return "\n".join(parts)

def _last_route_args(messages):
    """Find the coordinates of the last route tool call in the agent trace."""
    for m in reversed(messages):
        for call in getattr(m, "tool_calls", []) or []:
            if call.get("name") == "route":
                return call.get("args", {})
    return None

def _base_map():
    """Plain NYC map (Esri tiles), shown before any route is computed."""
    m = folium.Map(location=[40.73, -73.94], zoom_start=11, tiles=None)
    # OSM default tiles 403 in some environments; CartoDB positron now needs
    # an API key. Esri World Street Map is free, no key.
    folium.TileLayer(
        tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Street_Map/MapServer/tile/{z}/{y}/{x}",
        attr="Esri",
        name="Esri World Street Map",
    ).add_to(m)
    return m

def _route_map(origin_lat, origin_lon, dest_lat, dest_lon):
    """Fetch route geometry from OSRM and draw it on the NYC map."""
    url = f"https://router.project-osrm.org/route/v1/driving/{origin_lon},{origin_lat};{dest_lon},{dest_lat}"
    data = requests.get(url, params={"overview": "full", "geometries": "geojson"}, timeout=20).json()
    leg = data["routes"][0]
    coords = [(lat, lon) for lon, lat in leg["geometry"]["coordinates"]]
    m = _base_map()
    m.location = [(origin_lat + dest_lat) / 2, (origin_lon + dest_lon) / 2]
    folium.PolyLine(coords, color="blue", weight=4).add_to(m)
    folium.Marker((origin_lat, origin_lon), popup="Origin").add_to(m)
    folium.Marker((dest_lat, dest_lon), popup="Destination").add_to(m)
    return m

@st.cache_resource
def build_agent():
    model = ChatGoogleGenerativeAI(
        model="gemini-3.5-flash-lite",
        google_api_key=_get_api_key(),
        temperature=0,
    )
    return create_agent(
        model, [geocode, route],
        checkpointer=MemorySaver(),
        interrupt_before=["tools"],
        system_prompt=(
            "You are a New York City driving assistant. "
            "Always use the geocode tool to find the coordinates of the places "
            "and the route tool to compute driving time and distance. "
            "Never guess travel times from memory; report only numbers the tools returned."
        ),
    )

agent = build_agent()

st.title("GeoAI Agent — NYC Q&A")
st.caption("Ask about driving in New York City. The agent geocodes places and computes routes itself.")

if "messages" not in st.session_state:
    st.session_state.messages = []
if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())
if "last_route" not in st.session_state:
    st.session_state.last_route = None

# Layout: the map IS the platform (on top), the chat sits below it.
# The map slot is filled at the end of the script, so it already includes
# the route from the question just asked.
map_slot = st.empty()

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
        answer = _message_text(result["messages"][-1])
        st.markdown(answer)
        route_args = _last_route_args(result["messages"])
        if route_args:
            st.session_state.last_route = route_args
    st.session_state.messages.append({"role": "assistant", "content": answer})

# Draw the map into the top slot: the latest route, or a plain NYC map.
with map_slot:
    try:
        if st.session_state.last_route:
            r = st.session_state.last_route
            st_folium(
                _route_map(r["origin_lat"], r["origin_lon"], r["dest_lat"], r["dest_lon"]),
                width=1000, height=550,
            )
        else:
            st_folium(_base_map(), width=1000, height=550)
    except Exception:
        pass  # map is a bonus; the text answer already stands
