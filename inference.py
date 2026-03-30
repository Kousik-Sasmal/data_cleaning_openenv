"""
Inference Script
==============================
MANDATORY
- Before submitting, ensure the following variables are defined in your environment configuration:
    API_BASE_URL   The API endpoint for the LLM.
    MODEL_NAME     The model identifier to use for inference.
    HF_TOKEN       Your Hugging Face / API key.
    
- The inference script must be named `inference.py` and placed in the root directory of the project
- Participants must use OpenAI Client for all LLM calls using above variables
"""

import os
import json
import asyncio
from openai import AsyncOpenAI
from dotenv import load_dotenv

from models import DataCleaningAction
from client import DataCleaningEnv

load_dotenv()

API_BASE_URL = os.getenv("API_BASE_URL", "https://api.groq.com/openai/v1")
API_KEY = os.getenv("HF_TOKEN") or os.getenv("API_KEY")
MODEL_NAME = os.getenv("MODEL_NAME", "llama-3.1-8b-instant")

client = AsyncOpenAI(
    base_url=API_BASE_URL,
    api_key=API_KEY
)

async def _run_task(task_id: int):
    action_schema = DataCleaningAction.model_json_schema()

    async with DataCleaningEnv(base_url="http://localhost:8000") as env:
    # async with DataCleaningEnv(base_url="https://kousiksasmal-data-cleaning-env.hf.space") as env:
        obs_res = await env.reset(task_id=task_id)
        obs = obs_res.observation
        
        while not obs.done:
            sys_prompt = f"""You are an expert Data Cleaning Agent. 
            Your goal is to manipulate the dataset using precise commands to achieve the exact target format.
            Available commands: drop_duplicates, fill_na, format_date, filter, submit.
            You must output a single JSON object matching this JSON Schema: {json.dumps(action_schema)}
            """
            
            user_prompt = f"""
            Dataset Preview: {obs.dataset_preview}
            Available Columns: {obs.schema_info}
            Last System Message: {obs.message}
            
            What is your next action? Return ONLY the raw JSON object. Use double quotes.
            """
            
            try:
                completion = await client.chat.completions.create(
                    model=MODEL_NAME,
                    messages=[
                        {"role": "system", "content": sys_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    response_format={"type": "json_object"}
                )
                
                action_json = completion.choices[0].message.content
                print(f"[{task_id}] Action generated: {action_json}")
                
                action_dict = json.loads(action_json)
                action = DataCleaningAction(**action_dict)
                res = await env.step(action)
                obs = res.observation
                print(f"[{task_id}] System reply: {obs.message} | Score: {obs.current_score}")
                
            except Exception as e:
                print(f"[{task_id}] Execution error: {e}")
                res = await env.step(DataCleaningAction(command="submit", params={}))
                obs = res.observation
                break
                
        return obs.current_score

async def main():
    if not API_KEY:
        print("WARNING: HF_TOKEN or API_KEY is not set in environment. Inference WILL fail.")
        
    scores = []
    print(f"Starting Inference using {MODEL_NAME} at {API_BASE_URL}...")
    
    try:
        from server.tasks import TASKS
        num_tasks = len(TASKS)
    except Exception:
        num_tasks = 3
        
    for i in range(num_tasks):
        print(f"\n--- Running Task {i} ---")
        try:
            score = await _run_task(i)
            print(f"Task {i} Final Grader Score: {score}")
            scores.append(score)
        except Exception as e:
            print(f"Task {i} Inference Failed: {e}")
            scores.append(0.0)
            
    print(f"\nFinal Scores: {json.dumps(scores)}")
    return scores

if __name__ == "__main__":
    asyncio.run(main())
