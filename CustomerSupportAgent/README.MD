📞 24/7 AI Customer Support Application
This project provides a robust and intelligent 24/7 customer support system powered by LangGraph for workflow orchestration and Streamlit for an interactive web user interface. It automatically categorizes customer queries, analyzes sentiment, and provides appropriate responses or escalates negative interactions to a human agent.

✨ Features
Intelligent Query Categorization: Automatically classifies customer queries into "Technical," "Billing," or "General" categories.
Sentiment Analysis: Determines the sentiment of each query ("Positive," "Neutral," or "Negative") to tailor responses.
Automated Responses: Provides AI-generated answers for routine technical, billing, and general inquiries.
Human Escalation: Automatically flags and escalates queries with negative sentiment to a human agent for personalized support.
Interactive Web UI: A user-friendly interface built with Streamlit allows customers to submit queries and view instant AI responses.
🚀 Technologies Used
LangGraph: For defining the stateful, cyclic graphs that orchestrate the AI workflow.
LangChain: Core components for building language model applications.
OpenAI: Powers the large language models used for categorization, sentiment analysis, and response generation.
Streamlit: For creating the interactive and responsive web application UI.
python-dotenv: For securely managing environment variables.
🛠️ Setup and Installation
Follow these steps to get the project up and running on your local machine.

Prerequisites
Python 3.8+
An OpenAI API Key
1. Clone the Repository (Optional, if you're not using Git)

# Create a virtual environment
python -m venv venv

# Activate the virtual environment
# On Windows
.\venv\Scripts\activate
# On macOS/Linux
source venv/bin/activate

# Install the required Python packages
pip install -r requirements.txt

📞 24/7 AI Customer Support Application
This project provides a robust and intelligent 24/7 customer support system powered by LangGraph for workflow orchestration and Streamlit for an interactive web user interface. It automatically categorizes customer queries, analyzes sentiment, and provides appropriate responses or escalates negative interactions to a human agent.

✨ Features
Intelligent Query Categorization: Automatically classifies customer queries into "Technical," "Billing," or "General" categories.
Sentiment Analysis: Determines the sentiment of each query ("Positive," "Neutral," or "Negative") to tailor responses.
Automated Responses: Provides AI-generated answers for routine technical, billing, and general inquiries.
Human Escalation: Automatically flags and escalates queries with negative sentiment to a human agent for personalized support.
Interactive Web UI: A user-friendly interface built with Streamlit allows customers to submit queries and view instant AI responses.
🚀 Technologies Used
LangGraph: For defining the stateful, cyclic graphs that orchestrate the AI workflow.
LangChain: Core components for building language model applications.
OpenAI: Powers the large language models used for categorization, sentiment analysis, and response generation.
Streamlit: For creating the interactive and responsive web application UI.
python-dotenv: For securely managing environment variables.
🛠️ Setup and Installation
Follow these steps to get the project up and running on your local machine.

Prerequisites
Python 3.8+
An OpenAI API Key
1. Clone the Repository (Optional, if you're not using Git)
Bash

git clone <repository-url>
cd <project-directory>
2. Install Dependencies
It's recommended to use a virtual environment.

Bash

# Create a virtual environment
python -m venv venv

# Activate the virtual environment
# On Windows
.\venv\Scripts\activate
# On macOS/Linux
source venv/bin/activate

# Install the required Python packages
pip install -r requirements.txt
If you don't have a requirements.txt file, create one with the following content and then run the pip install command:

streamlit
langgraph
langchain-core
langchain_google_genai
python-dotenv

3. Configure Environment Variables
Create a file named .env in the root directory of your project and add your OpenAI API key:

OPENAI_API_KEY='your_openai_api_key_here'


▶️ How to Run
streamlit run app.py