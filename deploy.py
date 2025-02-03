import streamlit as st
import io
import sys
import os
import re
from phi.agent import Agent
from phi.model.groq import Groq
from phi.tools.duckduckgo import DuckDuckGo
from phi.tools.wikipedia import WikipediaTools
from phi.tools.googlesearch import GoogleSearch
import phi

def remove_ansi_escape_sequences(text):
    # Remove ANSI escape sequences (colors and other formatting)
    ansi_escape = re.compile(r'(?:\x1B[@-_][0-?]*[ -/]*[@-~])')
    return ansi_escape.sub('', text)

# Sidebar: Collect API keys from the user
st.sidebar.header("API Keys")
phi_api_key = st.sidebar.text_input("Enter PHI API Key", type="password")
groq_api_key = st.sidebar.text_input("Enter GROQ API Key", type="password")

if not phi_api_key or not groq_api_key:
    st.sidebar.warning("Please enter both API keys to continue.")
    st.stop()

# Set the API keys
phi.api = phi_api_key
os.environ["GROQ_API_KEY"] = groq_api_key

# Define your agents
web_search_agent = Agent(
    name="Healthcare Web Search Agent",
    description="Provides healthcare information from web resources.",
    role="Search the web for symptoms, conditions, and treatments.",
    model=Groq(id="deepseek-r1-distill-llama-70b"),
    tools=[DuckDuckGo()],
    instructions=[
        "Search for reliable healthcare information.",
        "Focus on symptoms, causes, and treatments."
    ],
    show_tool_calls=True,
    markdown=True,
)

wikipedia_agent = Agent(
    name="Healthcare Wikipedia Agent",
    description="Fetches healthcare information from Wikipedia.",
    model=Groq(id="deepseek-r1-distill-llama-70b"),
    tools=[WikipediaTools()],
    instructions=[
        "Provide accurate healthcare information from Wikipedia.",
        "Focus on medical conditions and treatments."
    ],
    show_tool_calls=True,
    markdown=True,
)

google_search_agent = Agent(
    name="Healthcare Google Search Agent",
    description="Searches Google for healthcare-related information.",
    model=Groq(id="deepseek-r1-distill-llama-70b"),
    tools=[GoogleSearch()],
    instructions=[
        "Retrieve healthcare information from trusted sources.",
        "Focus on medical conditions, symptoms, and treatments."
    ],
    show_tool_calls=True,
    debug_mode=True,
)

multi_ai_agent = Agent(
    name="Healthcare Chatbot Agent",
    description="Answers healthcare questions and provides reliable information.",
    team=[web_search_agent, wikipedia_agent, google_search_agent],
    model=Groq(id="deepseek-r1-distill-llama-70b"),
    instructions=[
        "Provide concise and accurate healthcare information.",
        "Use reliable sources only."
    ],
    show_tool_calls=True,
    markdown=True,
)

# Main UI
st.title("Healthcare Chatbot")
st.write("Ask your healthcare related questions below:")

user_query = st.text_input("Enter your question:", value="How to treat influenza at home?")

if st.button("Submit"):
    st.write("Processing...")
    placeholder = st.empty()
    
    # Capture the printed streaming output
    buffer = io.StringIO()
    old_stdout = sys.stdout
    sys.stdout = buffer
    try:
        # This function prints the response (stream=True prints chunks)
        multi_ai_agent.print_response(user_query, stream=True)
    except Exception as e:
        st.error(f"An error occurred: {e}")
    finally:
        sys.stdout = old_stdout

    # Get the captured output and remove any ANSI escape codes
    output_text = buffer.getvalue()
    clean_text = remove_ansi_escape_sequences(output_text)

    # Display the clean text in a markdown code block for clear formatting
    st.markdown(f"```\n{clean_text}\n```")
