import streamlit as st
from typing import Dict, TypedDict
from langgraph.graph import StateGraph, END
from langchain_core.prompts import ChatPromptTemplate
# --- IMPORTANT CHANGE: Import Google's Generative AI model ---
from langchain_google_genai import ChatGoogleGenerativeAI 
from dotenv import load_dotenv
import os
from model.State import State

# Load environment variables
load_dotenv()

# --- IMPORTANT CHANGE: Use Google's API key ---
os.environ["GOOGLE_API_KEY"] = os.getenv('GOOGLE_API_KEY') # Use your Google API key variable



def categorize(state: State) -> State:
    """Categorize the customer query into Technical, Billing, or General."""
    prompt = ChatPromptTemplate.from_template(
        "Categorize the following customer query into one of these categories: "
        "Technical, Billing, General. Query: {query}"
    )
    # --- IMPORTANT CHANGE: Instantiate Google's LLM ---
    chain = prompt | ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0) # Specify a Gemini model
    category = chain.invoke({"query": state["query"]}).content
    return {"category": category}

def analyze_sentiment(state: State) -> State:
    """Analyze the sentiment of the customer query as Positive, Neutral, or Negative."""
    prompt = ChatPromptTemplate.from_template(
        "Analyze the sentiment of the following customer query. "
        "Respond with either 'Positive', 'Neutral', or 'Negative'. Query: {query}"
    )
    # --- IMPORTANT CHANGE: Instantiate Google's LLM ---
    chain = prompt | ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0)
    sentiment = chain.invoke({"query": state["query"]}).content
    return {"sentiment": sentiment}

def handle_technical(state: State) -> State:
    """Provide a technical support response to the query."""
    prompt = ChatPromptTemplate.from_template(
        "Provide a technical support response to the following query: {query}"
    )
    # --- IMPORTANT CHANGE: Instantiate Google's LLM ---
    chain = prompt | ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0)
    response = chain.invoke({"query": state["query"]}).content
    return {"response": response}

def handle_billing(state: State) -> State:
    """Provide a billing support response to the query."""
    prompt = ChatPromptTemplate.from_template(
        "Provide a billing support response to the following query: {query}"
    )
    # --- IMPORTANT CHANGE: Instantiate Google's LLM ---
    chain = prompt | ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0)
    response = chain.invoke({"query": state["query"]}).content
    return {"response": response}

def handle_general(state: State) -> State:
    """Provide a general support response to the query."""
    prompt = ChatPromptTemplate.from_template(
        "Provide a general support response to the following query: {query}"
    )
    # --- IMPORTANT CHANGE: Instantiate Google's LLM ---
    chain = prompt | ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0)
    response = chain.invoke({"query": state["query"]}).content
    return {"response": response}

def escalate(state: State) -> State:
    """Escalate the query to a human agent due to negative sentiment."""
    return {"response": "This query has been escalated to a human agent due to its negative sentiment."}

def route_query(state: State) -> str:
    """Route the query based on its sentiment and category."""
    if state["sentiment"].strip() == "Negative": # Added .strip() for robustness
        return "escalate"
    elif state["category"].strip() == "Technical": # Added .strip() for robustness
        return "handle_technical"
    elif state["category"].strip() == "Billing": # Added .strip() for robustness
        return "handle_billing"
    else:
        return "handle_general"
    

