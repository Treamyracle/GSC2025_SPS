import os
import re
import json
from flask import Flask, request, jsonify
import vertexai
from crewai import Agent, Task, Crew
from langchain_google_vertexai import ChatVertexAI
import litellm
import google.auth
from google.auth.transport.requests import Request

app = Flask(__name__)

# ——— Konfigurasi project & kredensial ———
PROJECT_ID = os.getenv("GOOGLE_CLOUD_PROJECT") \
    or os.getenv("PROJECT_ID") \
    or "cool-state-453106-d5"
LOCATION = os.getenv("LOCATION", "us-central1")

os.environ.update({
    "PROJECT_ID": PROJECT_ID,
    "GOOGLE_CLOUD_PROJECT": PROJECT_ID,
    "VERTEX_PROJECT": PROJECT_ID,
    "VERTEX_LOCATION": LOCATION,
    "LITELLM_PROJECT_ID": PROJECT_ID,
    "LITELLM_LOCATION": LOCATION,
})

# Init Vertex AI & LiteLLM
litellm.set_verbose = True
vertexai.init(project=PROJECT_ID, location=LOCATION)

creds, _ = google.auth.default()
if creds.expired:
    creds.refresh(Request())
litellm.vertex_credentials = creds

def create_gemini_llm():
    # refresh creds per instance
    creds, _ = google.auth.default()
    if creds.expired:
        creds.refresh(Request())
    litellm.vertex_credentials = creds
    return ChatVertexAI(
        model_name="gemini-2.0-flash",
        project=PROJECT_ID,
        location=LOCATION
    )

# ——— Definisi Agents & Tasks ———
def get_destination_researcher():
    return Agent(
        role="Destination Researcher",
        goal=("For each city in the planned route, find the top 3–5 must-see attractions, "
              "including name, brief description, and recommended visit duration."),
        backstory=("You are a travel expert. Given a city name, you know its best landmarks, "
                   "museums, parks, and cultural sites. Provide details and how long to visit."),
        allow_delegation=False,
        verbose=True,
        llm=create_gemini_llm(),
    )

def get_route_planner():
    return Agent(
        role="Route Planner",
        goal=("Sequence the user's stops in a logical, one-way path per country."),
        backstory=("Based on the list of cities in each country, order them efficiently "
                   "(shortest-path, no back-tracking); decide border crossings; suggest airports."),
        allow_delegation=False,
        verbose=True,
        llm=create_gemini_llm(),
    )

def get_transport_agent():
    return Agent(
        role="Transport Planner",
        goal=("Determine transport modes, durations, and connections between each leg."),
        backstory=("For each leg: calculate transit time, choose mode (bus/train/flight), "
                   "slot real flights if provided, else propose approximate times."),
        allow_delegation=False,
        verbose=True,
        llm=create_gemini_llm(),
    )

def get_itinerary_writer():
    return Agent(
        role="Itinerary Writer",
        goal=("Compose a day-by-day itinerary in Markdown."),
        backstory=("Using route and transport plans: break into days; list activities with times, "
                   "locations, transport legs; highlight travel days."),
        allow_delegation=False,
        verbose=True,
        llm=create_gemini_llm(),
    )

def get_itinerary_parser():
    return Agent(
        role="Itinerary Parser",
        goal=("Extract and structure itinerary data from markdown and route information."),
        backstory=("Expert at parsing travel itineraries into clean JSON with city/checkin/checkout."),
        allow_delegation=False,
        verbose=True,
        llm=create_gemini_llm(),
    )

def get_plan_route_task():
    return Task(
        description=("Given trip parameters {countries}, {arrival}, {departure}, {travelers}: "
                     "1. List key cities per country. 2. Order them one-way. 3. Decide crossings."),
        expected_output="An ordered list of stops per country.",
        agent=get_route_planner(),
        output_key="route",
    )

def get_research_destinations_task():
    return Task(
        description=("Here is the planned route {route}. For each city, find top 3–5 attractions "
                     "with name, 1-sentence description, estimated visit time."),
        expected_output="A list of {'city':…, 'attractions':[…]} entries.",
        agent=get_destination_researcher(),
        output_key="attractions",
    )

def get_plan_transport_task():
    return Task(
        description=("Given the route {route} and inputs, plan transport legs with mode, duration, times."),
        expected_output="A list of transport segments.",
        agent=get_transport_agent(),
        output_key="transport_segments",
    )

def get_write_itinerary_task():
    return Task(
        description=("Using {route}, {transport_segments}, {attractions}, produce a Markdown itinerary."),
        expected_output="A full Markdown itinerary.",
        agent=get_itinerary_writer(),
        output_key="itinerary_md",
    )

def get_parse_itinerary_task():
    return Task(
        description=("Given {route} and Markdown {itinerary_md}, extract JSON array of {city,checkin,checkout}."),
        expected_output="A simple JSON array with those fields.",
        agent=get_itinerary_parser(),
        output_key="parsed_itinerary",
    )

# ——— JSON Parser untuk itinerary_data ———
def parse_itinerary_json_string(raw_string: str):
    # Bersihkan markdown fences
    cleaned = raw_string.replace('```json', '').replace('```', '')
    # Unescape \n and \"
    cleaned = cleaned.encode('utf-8').decode('unicode_escape')
    # Parse JSON
    return json.loads(cleaned)

# ——— Routes ———
@app.route("/", methods=["GET"])
def health_check():
    return jsonify({"message": "Cloud Run service is up!", "project_id": PROJECT_ID})

@app.route("/run", methods=["POST"])
def generate_itinerary():
    inputs = request.get_json(force=True)
    try:
        # 1) Route
        route = Crew(
            agents=[get_route_planner()],
            tasks=[get_plan_route_task()],
            manager_llm=create_gemini_llm(),
            project_id=PROJECT_ID, location=LOCATION
        ).kickoff(inputs=inputs).raw

        # 2) Destinations
        attractions = Crew(
            agents=[get_destination_researcher()],
            tasks=[get_research_destinations_task()],
            manager_llm=create_gemini_llm(),
            project_id=PROJECT_ID, location=LOCATION
        ).kickoff(inputs={"route": route}).raw

        # 3) Transport
        transport = Crew(
            agents=[get_transport_agent()],
            tasks=[get_plan_transport_task()],
            manager_llm=create_gemini_llm(),
            project_id=PROJECT_ID, location=LOCATION
        ).kickoff(inputs={**inputs, "route": route}).raw

        # 4) Itinerary Markdown
        itinerary_md = Crew(
            agents=[get_itinerary_writer()],
            tasks=[get_write_itinerary_task()],
            manager_llm=create_gemini_llm(),
            project_id=PROJECT_ID, location=LOCATION
        ).kickoff(inputs={
            "route": route,
            "attractions": attractions,
            "transport_segments": transport
        }).raw

        # 5) Parse itinerary to JSON
        parsed = parse_itinerary_json_string(itinerary_md)

        return jsonify({
            "route": route,
            "attractions": attractions,
            "transport": transport,
            "itinerary_markdown": itinerary_md,
<<<<<<< HEAD
            "itinerary_data": parsed_to_json,
            "pre_parsed": parsed_itinerary
            
=======
            "itinerary_data": parsed
>>>>>>> 7a9029a908ed54e990e566ad9ebe14ea2016338a
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    port = int(os.getenv("PORT", "8080"))
    app.run(host="0.0.0.0", port=port)
