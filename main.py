import os
from flask import Flask, request, jsonify
import vertexai
from crewai import Agent, Task, Crew
from langchain_google_vertexai import ChatVertexAI
import litellm
import google.auth
from google.auth import credentials
from google.auth.transport.requests import Request
import json

app = Flask(__name__)

# Inisialisasi PROJECT_ID dengan nilai default
PROJECT_ID = "cool-state-453106-d5"  # Default value
LOCATION = os.getenv("LOCATION", "us-central1")

# Coba ambil dari environment variable jika ada
env_project_id = (
    os.getenv("GOOGLE_CLOUD_PROJECT") or
    os.getenv("PROJECT_ID") or
    os.getenv("CLOUDSDK_CORE_PROJECT") or
    os.getenv("project_id")
)

if env_project_id:
    PROJECT_ID = env_project_id
    print("Using PROJECT_ID from environment:", PROJECT_ID)
else:
    print("Using default PROJECT_ID:", PROJECT_ID)

# Coba ambil dari metadata server jika di Cloud Run
try:
    import requests
    metadata_url = "http://metadata.google.internal/computeMetadata/v1/project/project-id"
    headers = {"Metadata-Flavor": "Google"}
    response = requests.get(metadata_url, headers=headers, timeout=1)
    if response.status_code == 200:
        PROJECT_ID = response.text
        print("Using PROJECT_ID from metadata server:", PROJECT_ID)
except:
    print("Could not get PROJECT_ID from metadata server, using:", PROJECT_ID)

# Set environment variable untuk memastikan library lain bisa mengaksesnya
os.environ["PROJECT_ID"] = PROJECT_ID
os.environ["GOOGLE_CLOUD_PROJECT"] = PROJECT_ID
os.environ["VERTEX_PROJECT"] = PROJECT_ID
os.environ["VERTEX_LOCATION"] = LOCATION
os.environ["LITELLM_PROJECT_ID"] = PROJECT_ID
os.environ["LITELLM_LOCATION"] = LOCATION

# Konfigurasi LiteLLM
litellm.set_verbose = True

# Initialize the Vertex AI SDK using Application Default Credentials
vertexai.init(project=PROJECT_ID, location=LOCATION)

# Dapatkan credentials dan project_id
try:
    creds, project = google.auth.default()
    if creds.expired:
        creds.refresh(Request())
    print("Credentials refreshed successfully")
    
    # Set credentials untuk LiteLLM
    litellm.vertex_credentials = creds
    litellm.vertex_project = project or PROJECT_ID
    litellm.vertex_location = LOCATION
    
    # Set environment variables tambahan
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = json.dumps(creds.to_json())
    os.environ["VERTEX_CREDENTIALS"] = json.dumps(creds.to_json())
    
    print("LiteLLM configured with project_id:", litellm.vertex_project)
except Exception as e:
    print(f"Error setting up credentials: {e}")

def create_gemini_llm():
    """Create a new instance of ChatVertexAI with current project settings"""
    # Pastikan environment variables tersedia untuk setiap instance
    os.environ["PROJECT_ID"] = PROJECT_ID
    os.environ["GOOGLE_CLOUD_PROJECT"] = PROJECT_ID
    os.environ["VERTEX_PROJECT"] = PROJECT_ID
    os.environ["VERTEX_LOCATION"] = LOCATION
    os.environ["LITELLM_PROJECT_ID"] = PROJECT_ID
    os.environ["LITELLM_LOCATION"] = LOCATION
    
    # Refresh credentials untuk setiap instance
    try:
        creds, project = google.auth.default()
        if creds.expired:
            creds.refresh(Request())
        
        # Set credentials untuk LiteLLM
        litellm.vertex_credentials = creds
        litellm.vertex_project = project or PROJECT_ID
        litellm.vertex_location = LOCATION
        
        # Set environment variables tambahan
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = json.dumps(creds.to_json())
        os.environ["VERTEX_CREDENTIALS"] = json.dumps(creds.to_json())
        
        print("LiteLLM instance configured with project_id:", litellm.vertex_project)
    except Exception as e:
        print(f"Error refreshing credentials in create_gemini_llm: {e}")
    
    return ChatVertexAI(
        model_name="gemini-2.0-flash",
        project=PROJECT_ID,
        location=LOCATION
    )

# Instantiate Gemini LLM via LangChain/Vertex AI
gemini_flash = create_gemini_llm()

def reinit_agent(agent):
    """Reinitialize an agent with fresh LLM instance"""
    agent.llm = create_gemini_llm()
    return agent

# 1. Define your Agents with function to reinitialize them
def get_destination_researcher():
    return Agent(
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
        verbose=False,
        llm=create_gemini_llm(),
    )

def get_input_agent():
    return Agent(
        role="User Input Collector",
        goal="Gather all trip parameters from the user",
        backstory=(
            "You ask the user for:\n"
            "- One or more countries they want to visit\n"
            "- Arrival airport, date & time\n"
            "- Departure airport, date & time (if known)\n"
            "- Number of travelers, with ages for any under 18\n"
            "- If they don't yet have flights, only dates for departure & return\n"
            "Collect and validate this into a structured data format."
        ),
        allow_delegation=False,
        verbose=False,
        llm=create_gemini_llm(),
    )

def get_route_planner():
    return Agent(
        role="Route Planner",
        goal="Sequence the user's stops in a logical, one-way path per country",
        backstory=(
            "Based on the list of cities in each country,\n"
            "- Order them so travel is efficient (e.g. shortest-path, no back-tracking)\n"
            "- If multiple countries, decide border crossings (e.g. bus, train or flight)\n"
            "- Suggest nearest airports for inter-country flights or buses."
        ),
        allow_delegation=False,
        verbose=False,
        llm=create_gemini_llm(),
    )

def get_transport_agent():
    return Agent(
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
        verbose=False,
        llm=create_gemini_llm(),
    )

def get_itinerary_writer():
    return Agent(
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
        verbose=False,
        llm=create_gemini_llm(),
    )

def get_itinerary_parser():
    return Agent(
        role="Itinerary Parser",
        goal="Extract and structure itinerary data from markdown and route information",
        backstory=(
            "You are an expert at parsing travel itineraries and extracting structured data. "
            "You can identify cities, dates, and activities from various formats and present them "
            "in a clean, organized JSON structure."
        ),
        allow_delegation=False,
        verbose=False,
        llm=create_gemini_llm(),
    )

# Define tasks with functions instead of direct agent references
def get_plan_route_task():
    return Task(
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
            "**OUTPUT FORMAT**: ONLY return a JSON array. No commentary, no markdown, no extra keys."
        ),
        expected_output="An ordered list of stops per country, plus any inter-country legs.",
        agent=get_route_planner(),
        output_key="route",
    )

def get_research_destinations_task():
    return Task(
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
        agent=get_destination_researcher(),
        output_key="attractions",
    )

def get_plan_transport_task():
    return Task(
        description=(
            "Given the route:\n"
            "{route}\n\n"
            "1. For each leg between stops, calculate distance & suggest mode "
            "(train, bus, flight).\n"
            "2. Estimate departure & arrival times based on the user's schedule.\n"
            "3. If the user supplied real flights, slot them in."
        ),
        expected_output="A list of transport segments with mode, duration, and times.",
        agent=get_transport_agent(),
        output_key="transport_segments",
    )

def get_write_itinerary_task():
    return Task(
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
        agent=get_itinerary_writer(),
        output_key="itinerary_md",
    )

def get_parse_itinerary_task():
    return Task(
        description=(
            "Given the following data:\n"
            "- Route: {route}\n"
            "- Itinerary Markdown: {itinerary_md}\n\n"
            "Extract and structure the following information:\n"
            "1. For each city in the itinerary:\n"
            "   - City name (extract only the city name, remove any additional text like dates)\n"
            "   - Check-in date (format: MM/DD, e.g., '07/01' for July 1)\n"
            "   - Check-out date (format: MM/DD, e.g., '07/02' for July 2)\n"
            "2. Format the output as a JSON array of objects with the following structure:\n"
            "[\n"
            "  {\n"
            "    \"city\": \"city name\",\n"
            "    \"checkin\": \"MM/DD\",\n"
            "    \"checkout\": \"MM/DD\"\n"
            "  },\n"
            "  ...\n"
            "]\n"
            "3. Skip any entries that are not actual cities (such as 'Border Crossing' or country names)\n"
            "4. Ensure dates are properly formatted and consistent in MM/DD format\n\n"
            "Example expected output:\n"
            "[\n"
            "  {\"city\": \"Bangkok\", \"checkin\": \"07/01\", \"checkout\": \"07/03\"},\n"
            "  {\"city\": \"Chiang Mai\", \"checkin\": \"07/03\", \"checkout\": \"07/06\"},\n"
            "  {\"city\": \"Luang Prabang\", \"checkin\": \"07/06\", \"checkout\": \"07/08\"}\n"
            "]"
            "5.**OUTPUT FORMAT**: ONLY return a JSON array. No commentary, no markdown, no extra keys."
        ),
        expected_output="A JSON array containing city and date information with MM/DD date format",
        agent=get_itinerary_parser(),
        output_key="itinerary_data",
    )

@app.route("/", methods=["GET"])
def health_check():
    return jsonify({
        "message": "🌟 Cloud Run service is up!",
        "project_id": PROJECT_ID
    }), 200

@app.route("/run", methods=["POST"])
def generate_itinerary():
    print("GOOGLE_CLOUD_PROJECT:", os.getenv("GOOGLE_CLOUD_PROJECT"))
    print("PROJECT_ID:", os.getenv("PROJECT_ID"))
    print("CLOUDSDK_CORE_PROJECT:", os.getenv("CLOUDSDK_CORE_PROJECT"))
    print("project_id:", os.getenv("project_id"))
    print("Final PROJECT_ID used:", PROJECT_ID)

    inputs = request.get_json(force=True)

    # Pastikan PROJECT_ID selalu tersedia di environment
    current_project_id = PROJECT_ID
    os.environ["PROJECT_ID"] = current_project_id
    os.environ["GOOGLE_CLOUD_PROJECT"] = current_project_id

    try:
        # 1) Plan route dengan agent yang baru
        route_planner = get_route_planner()
        plan_route_task = get_plan_route_task()
        route_crew = Crew(
            agents=[route_planner],
            tasks=[plan_route_task],
            manager_llm=create_gemini_llm(),
            project_id=current_project_id,
            location=LOCATION
        )
        route_res = route_crew.kickoff(inputs=inputs).raw
        print("Route planning completed successfully")

        # 2) Research destinations dengan agent yang baru
        destination_researcher = get_destination_researcher()
        research_destinations_task = get_research_destinations_task()
        dest_crew = Crew(
            agents=[destination_researcher],
            tasks=[research_destinations_task],
            manager_llm=create_gemini_llm(),
            project_id=current_project_id,
            location=LOCATION
        )
        attractions = dest_crew.kickoff(inputs={"route": route_res}).raw
        print("Destination research completed successfully")

        # 3) Plan transport dengan agent yang baru
        transport_agent = get_transport_agent()
        plan_transport_task = get_plan_transport_task()
        trans_crew = Crew(
            agents=[transport_agent],
            tasks=[plan_transport_task],
            manager_llm=create_gemini_llm(),
            project_id=current_project_id,
            location=LOCATION
        )
        transport = trans_crew.kickoff(inputs={"route": route_res, **inputs}).raw
        print("Transport planning completed successfully")

        # 4) Write itinerary dengan agent yang baru
        itinerary_writer = get_itinerary_writer()
        write_itinerary_task = get_write_itinerary_task()
        write_crew = Crew(
            agents=[itinerary_writer],
            tasks=[write_itinerary_task],
            manager_llm=create_gemini_llm(),
            project_id=current_project_id,
            location=LOCATION
        )
        itinerary_md = write_crew.kickoff(inputs={
            "route": route_res,
            "attractions": attractions,
            "transport_segments": transport
        }).raw
        print("Itinerary writing completed successfully")

        # Add itinerary parsing step
        itinerary_parser = get_itinerary_parser()
        parse_itinerary_task = get_parse_itinerary_task()
        parse_crew = Crew(
            agents=[itinerary_parser],
            tasks=[parse_itinerary_task],
            manager_llm=create_gemini_llm(),
            project_id=current_project_id,
            location=LOCATION
        )

        itinerary_data = parse_crew.kickoff(inputs={
            "route": route_res,
            "itinerary_md": itinerary_md,
            "attractions": attractions
        }).raw
        
        return jsonify({
            "itinerary_markdown": itinerary_md,
            "pre_parsed": itinerary_data
        })
    
    except Exception as e:
        print(f"Error occurred: {str(e)}")
        return jsonify({
            "error": str(e),
            "project_id": current_project_id,
            "env_project_id": os.environ.get("PROJECT_ID"),
            "env_google_cloud_project": os.environ.get("GOOGLE_CLOUD_PROJECT")
        }), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", "8080"))
    app.run(host="0.0.0.0", port=port)