import os
from dotenv import load_dotenv
from crewai import Agent, Task, Crew
from crewai import LLM
from crewai.tools import tool
from langchain_community.tools import DuckDuckGoSearchRun
import gradio as gr

os.environ["OPENAI_API_KEY"] = "not-used-but-required-by-crewai"

# ────────────────────────────────────────────────
#  1. Load API key (recommended: use .env file)
# ────────────────────────────────────────────────

load_dotenv()           # looks for .env in current folder
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not GROQ_API_KEY:
    GROQ_API_KEY = input("Please enter your Groq API key: ").strip()
    if not GROQ_API_KEY:
        raise ValueError("Groq API key is required.")

# ────────────────────────────────────────────────
#  2. Initialize LLM (using Groq)
# ────────────────────────────────────────────────

llm = LLM(
    model="groq/llama-3.3-70b-versatile",   # or llama-3.3-70b, gemma2-27b, etc.
    api_key=GROQ_API_KEY,
    temperature=0.7,
    max_tokens=1024,
)

# ────────────────────────────────────────────────
#  3. Define a simple search tool
# ────────────────────────────────────────────────



@tool("Web Search via DuckDuckGo")
def web_search(query: str) -> str:
    """A search engine tool to look up current information on the internet.
    Input MUST be a single, clear search query string (question or keywords).
    Useful when you need real-time facts, news, stats, or verification."""
    ddg = DuckDuckGoSearchRun()
    return ddg.run(query)

# ────────────────────────────────────────────────
#  4. Define Researcher Agent
# ────────────────────────────────────────────────

researcher = Agent(
    role="Web Researcher",
    goal="Find accurate, up-to-date information from the web to answer questions",
    backstory=(
        "You are a fast, skeptical researcher who loves finding primary sources "
        "and giving concise, factual answers. You only use the search tool when needed."
    ),
    verbose=True,
    allow_delegation=False,
    llm=llm,
    tools=[web_search],
)

# ────────────────────────────────────────────────
#  5. Define a simple task template
# ────────────────────────────────────────────────

def create_research_task(question: str) -> Task:
    return Task(
        description=(
            f"Research and answer the following question as factually and concisely as possible:\n\n"
            f"QUESTION: {question}\n\n"
            "Use web search only when necessary. Cite key sources briefly when you do. "
            "If you're very confident without searching, just answer directly."
        ),
        expected_output="A clear, concise answer in 3–8 sentences max + sources if used.",
        agent=researcher,
    )

# ────────────────────────────────────────────────
#  6. Gradio interface
# ────────────────────────────────────────────────

def run_research(question):
    if not question.strip():
        return "Please enter a question."
    
    try:
        task = create_research_task(question)
        crew = Crew(
            agents=[researcher],
            tasks=[task],
            verbose=True,              
        )
        result = crew.kickoff()
        return str(result)
    except Exception as e:
        return f"Error: {str(e)}"

# ────────────────────────────────────────────────
#  7. Launch UI
# ────────────────────────────────────────────────

demo = gr.Interface(
    fn=run_research,
    inputs=gr.Textbox(
        label="Your question",
        placeholder="e.g. What is the current status of Grok-3 development?",
        lines=2,
    ),
    outputs=gr.Textbox(label="Answer"),
    title="Simple Tool-Calling Research Agent",
    description="Ask anything — powered by CrewAI + Groq + DuckDuckGo",
    examples=[
        ["What was the score of the last Bangladesh vs India cricket match?"],
        ["Latest news about xAI Colossus cluster"],
        ["Who won the 2025 Ballon d'Or?"],
    ],
    flagging_mode="never",
)

if __name__ == "__main__":
    demo.launch(share=False)   # change to share=True if you want public link (temporary)