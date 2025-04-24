# main.py
import os
from flask import Flask, request, jsonify
import vertexai
from crewai import Agent, Task, Crew
from langchain_google_vertexai import ChatVertexAI

app = Flask(__name__)

# Cloud Run otomatis meng-set GOOGLE_CLOUD_PROJECT ke project ID-mu
PROJECT_ID = os.getenv("GOOGLE_CLOUD_PROJECT")
# Default LOCATION jika tidak di-set
LOCATION   = os.getenv("LOCATION", "us-central1")

# Inisialisasi SDK Vertex AI
vertexai.init(project=PROJECT_ID, location=LOCATION)

# Instansiasi Gemini via LangChain/Vertex
gemini_flash = ChatVertexAI(
    model_name="gemini-2.0-flash",
    project_id=PROJECT_ID,   # <-- gunakan project_id, bukan project
    location=LOCATION
)

# 1. Define your Agents
destination_researcher = Agent(
    role="Destination Researcher",
    goal=(
        "For each city in the planned route, "
        "find the top 3–5 must-see attractions, "
        "including name, brief description, "
        "and recommended visit duration."
    ),
    backstory=(
        "You are a travel expert. "
        "Given a city name, you know its best landmarks, "
        "museums, parks, and cultural sites. "
        "Provide details and how long it takes to visit each."
    ),
    allow_delegation=False,
    verbose=True,
    llm=gemini_flash,
)

input_agent = Agent(
    role="User Input Collector",
    goal="Gather all trip parameters from the user",
    backstory=(
        "You ask the user for:\n"
        "- One or more countries they want to visit\n"
        "- Arrival airport, date & time\n"
        "- Departure airport, date & time (if known)\n"
        "- Number of travelers, with ages for any under 18\n"
        "- If they don’t yet have flights, only dates for departure & return\n"
        "Collect and validate this into a structured data format."
    ),
    allow_delegation=False,
    verbose=True,
    llm=gemini_flash,
)

route_planner = Agent(
    role="Route Planner",
    goal="Sequence the user’s stops in a logical, one-way path per country",
    backstory=(
        "Based on the list of cities in each country,\n"
        "- Order them so travel is efficient (e.g. shortest-path, no back-tracking)\n"
        "- If multiple countries, decide border crossings (e.g. bus, train or flight)\n"
        "- Suggest nearest airports for inter-country flights or buses."
    ),
    allow_delegation=False,
    verbose=True,
    llm=gemini_flash,
)

transport_agent = Agent(
    role="Transport Planner",
    goal="Determine transport modes, durations, and connections between each leg",
    backstory=(
        "For each leg of the route:\n"
        "- Calculate transit time between locations (e.g. city A → city B)\n"
        "- Choose transport mode (bus, train, domestic flight) based on distance and speed\n"
        "- If user provided actual flight info, slot it in\n"
        "- Otherwise, propose approximate departure/arrival times."
    ),
    allow_delegation=False,
    verbose=True,
    llm=gemini_flash,
)

itinerary_writer = Agent(
    role="Itinerary Writer",
    goal="Compose a day-by-day itinerary in Markdown",
    backstory=(
        "Using the route and transport plans:\n"
        "- Break into Day 1, Day 2, … up to total trip days\n"
        "- For each day, list activities in order with:\n"
        "    • Start/end times\n"
        "    • Location names\n"
        "    • Transport mode & duration between stops\n"
        "- Highlight travel days separately when crossing countries."
    ),
    allow_delegation=False,
    verbose=True,
    llm=gemini_flash,
)

plan_route = Task(
    description=(
        "Given the trip parameters:\n"
        "- Countries to visit: {countries}\n"
        "- Arrival info: {arrival}\n"
        "- Departure info: {departure}\n"
        "- Travelers: {travelers}\n\n"
        "1. For each country, list the key cities the user wants to visit.\n"
        "2. Order them in a single-direction route "
        "(e.g. Geneva → Lausanne → Bern → Interlaken → Lucerne → Zurich).\n"
        "3. If multiple countries, decide border crossings (bus/train/flight) and nearest airports."
    ),
    expected_output="An ordered list of stops per country, plus any inter-country legs.",
    agent=route_planner,
    output_key="route",
)

research_destinations = Task(
    description=(
        "Here is the planned route:\n"
        "{route}\n\n"
        "For *each* city in that route, find the top 3–5 attractions:\n"
        "- Name\n"
        "- 1-sentence description\n"
        "- Estimated visit time (e.g. 1h, 2h)\n\n"
        "Return a JSON-style list of dicts."
    ),
    expected_output="A list of {'city':…, 'attractions':[…]} entries.",
    agent=destination_researcher,
    output_key="attractions",
)

plan_transport = Task(
    description=(
        "Given the route:\n"
        "{route}\n\n"
        "1. For each leg between stops, calculate distance & suggest mode "
        "(train, bus, flight).\n"
        "2. Estimate departure & arrival times based on the user’s schedule.\n"
        "3. If the user supplied real flights, slot them in."
    ),
    expected_output="A list of transport segments with mode, duration, and times.",
    agent=transport_agent,
    output_key="transport_segments",
)

write_itinerary = Task(
    description=(
        "Using the following data:\n"
        "- Route: {route}\n"
        "- Transport: {transport_segments}\n"
        "- Attractions: {attractions}\n\n"
        "Produce a day-by-day Markdown itinerary:\n"
        "• For each city: list attractions with their visit times.\n"
        "• For each travel leg: show departure/arrival times & mode."
    ),
    expected_output="A full Markdown itinerary incorporating attractions and travel details.",
    agent=itinerary_writer,
    output_key="itinerary_md",
)

@app.route("/", methods=["GET"])
def health_check():
    project_id = os.getenv("GOOGLE_CLOUD_PROJECT", "default-project-id")
    return jsonify({
        "message": "🌟 Cloud Run service is up!",
        "project_id": project_id
    }), 200

@app.route("/run", methods=["POST"])
def generate_itinerary():
    inputs = request.get_json(force=True)
    # 1) Plan route
    route_crew = Crew(
        agents=[route_planner],
        tasks=[plan_route],
        manager_llm=gemini_flash,
        project_id=PROJECT_ID, location=LOCATION
    )
    route_res = route_crew.kickoff(inputs=inputs).raw

    # 2) Research destinations
    dest_crew = Crew(
        agents=[destination_researcher],
        tasks=[research_destinations],
        manager_llm=gemini_flash,
        project_id=PROJECT_ID, location=LOCATION
    )
    attractions = dest_crew.kickoff(inputs={"route": route_res}).raw

    # 3) Plan transport
    trans_crew = Crew(
        agents=[transport_agent],
        tasks=[plan_transport],
        manager_llm=gemini_flash,
        project_id=PROJECT_ID, location=LOCATION
    )
    transport = trans_crew.kickoff(inputs={"route": route_res, **inputs}).raw

    # 4) Write itinerary
    write_crew = Crew(
        agents=[itinerary_writer],
        tasks=[write_itinerary],
        manager_llm=gemini_flash,
        project_id=PROJECT_ID, location=LOCATION
    )
    itinerary_md = write_crew.kickoff(inputs={
        "route": route_res,
        "attractions": attractions,
        "transport_segments": transport
    }).raw

    return jsonify({
        "route": route_res,
        "attractions": attractions,
        "transport": transport,
        "itinerary_markdown": itinerary_md
    })

if __name__ == "__main__":
    port = int(os.environ.get("PORT", "8080"))
    app.run(host="0.0.0.0", port=port)
