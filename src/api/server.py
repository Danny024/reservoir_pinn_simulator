from fastapi import FastAPI
from pydantic import BaseModel
import requests
import json
import torch
import torch.nn as nn
from dotenv import load_dotenv
import os
from src.models.pinn import PINN

# Load environment variables
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY", "")

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load PINN model
model = PINN().to(device)
try:
    model.load_state_dict(torch.load('reservoir_pinn.pth', map_location=device))
    model.eval()
except FileNotFoundError:
    print("Warning: reservoir_pinn.pth not found. Model is untrained.")

app = FastAPI()

class QueryRequest(BaseModel):
    prompt: str
    llm_model: str

# Function to query DeepSeek via Ollama
def query_deepseek(prompt):
    ollama_url = "http://localhost:11434/api/generate"
    payload = {
        "model": "deepseek-r1:1.5b",
        "prompt": f"""
        You are an assistant that extracts coordinates from user queries about pressure in a 3D reservoir model.
        The query will ask for pressure at specific x, y, z, t values. Extract the numerical values for x, y, z, and t
        and return them in JSON format. If the query is unclear or coordinates are missing, return an error message.
        Example input: "What is the pressure at x=0.5, y=0.5, z=0.5, t=0.5?"
        Example output: {{"x": 0.5, "y": 0.5, "z": 0.5, "t": 0.5}}
        Input: {prompt}
        """,
        "format": "json",
        "stream": False
    }
    try:
        response = requests.post(ollama_url, json=payload)
        if response.status_code == 200:
            return json.loads(response.json()['response'])
        else:
            return {"response": "Error querying DeepSeek"}
    except Exception as e:
        return {"response": f"DeepSeek error: {str(e)}"}

# Function to query OpenAI
def query_openai(prompt, api_key):
    openai_url = "https://api.openai.com/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": "gpt-4",
        "messages": [
            {
                "role": "system",
                "content": """
                You are an assistant that extracts coordinates from user queries about pressure in a 3D reservoir model.
                The query will ask for pressure at specific x, y, z, t values. Extract the numerical values for x, y, z, and t
                and return them in JSON format. If the query is unclear or coordinates are missing, return an error message.
                Example input: "What is the pressure at x=0.5, y=0.5, z=0.5, t=0.5?"
                Example output: {"x": 0.5, "y": 0.5, "z": 0.5, "t": 0.5}
                """
            },
            {
                "role": "user",
                "content": prompt
            }
        ],
        "response_format": {"type": "json_object"}
    }
    try:
        response = requests.post(openai_url, headers=headers, json=payload)
        if response.status_code == 200:
            return json.loads(response.json()['choices'][0]['message']['content'])
        else:
            return {"response": "Error querying OpenAI"}
    except Exception as e:
        return {"response": f"OpenAI error: {str(e)}"}

@app.post("/query")
async def query_pressure(request: QueryRequest):
    if request.llm_model == "OpenAI" and openai_api_key:
        result = query_openai(request.prompt, openai_api_key)
        if "response" in result and "error" in result["response"].lower():
            print("OpenAI query failed, falling back to DeepSeek")
            result = query_deepseek(request.prompt)
    else:
        result = query_deepseek(request.prompt)
    
    if all(key in result for key in ['x', 'y', 'z', 't']):
        try:
            x, y, z, t = float(result['x']), float(result['y']), float(result['z']), float(result['t'])
            inputs = torch.tensor([[x, y, z, t]], dtype=torch.float32).to(device)
            with torch.no_grad():
                pressure = model(inputs).item()
            return {"x": x, "y": y, "z": z, "t": t, "pressure": pressure}
        except Exception as e:
            return {"response": f"Error computing pressure: {str(e)}"}
    return result