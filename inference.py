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
import sys
import json
import asyncio
import re
from typing import Optional, List
from openai import AsyncOpenAI
from dotenv import load_dotenv

from models import DataCleaningAction
from client import DataCleaningEnv

load_dotenv()

API_BASE_URL = os.getenv("API_BASE_URL", "https://api.groq.com/openai/v1")
API_KEY = os.getenv("HF_TOKEN") or os.getenv("API_KEY")
MODEL_NAME = os.getenv("MODEL_NAME", "openai/gpt-oss-20b")
TASK_NAME = os.getenv("DATA_CLEANING_ENV_TASK", "data_cleaning_task_0")
BENCHMARK = os.getenv("DATA_CLEANING_ENV_BENCHMARK", "data_cleaning_env")
MAX_STEPS = 15

def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)

def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_val = error if error else "null"
    done_val = str(done).lower()
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} done={done_val} error={error_val}",
        flush=True,
    )

def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    # According to OpenEnv validator checks, standard might enforce .3f or .2f
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(f"[END] success={str(success).lower()} steps={steps} score={score:.3f} rewards={rewards_str}", flush=True)

async def main():
    # Initialize variables at the top so log_end can see them even if reset() fails
    steps_taken = 0
    rewards = []
    score = 0.0
    success = False
    last_score = 0.0
    
    if not API_KEY:
        print("WARNING: HF_TOKEN or API_KEY is not set in environment. Inference WILL fail.", file=sys.stderr)
        
    client = AsyncOpenAI(
        base_url=API_BASE_URL,
        api_key=API_KEY
    )
    
    # parse task_id from TASK_NAME if present (e.g. "data_cleaning_0" -> 0)
    try:
        task_id = int(TASK_NAME.split("_")[-1])
    except (ValueError, IndexError):
        task_id = 0
        
    action_schema = DataCleaningAction.model_json_schema()

    log_start(task=TASK_NAME, env=BENCHMARK, model=MODEL_NAME)
    
    try:
        async with DataCleaningEnv(base_url="http://localhost:8000") as env:
        # async with DataCleaningEnv(base_url="https://kousiksasmal-data-cleaning-env.hf.space") as env:
            obs_res = await env.reset(task_id=task_id)
            obs = obs_res.observation
            
            for _ in range(MAX_STEPS):
                if obs.done:
                    break
                    
                steps_taken += 1
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
                
                action_str_log = "null"
                error_msg = None
                step_reward = 0.0
                obs_done = False
                
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
                    # Clean newlines to keep the [STEP] line clean
                    action_str_log = re.sub(r'\s+', ' ', action_json).strip()
                    
                    action_dict = json.loads(action_json)
                    action = DataCleaningAction(**action_dict)
                    res = await env.step(action)
                    obs = res.observation
                    obs_done = obs.done
                    
                    # Partial progress calculation
                    step_reward = max(0.0, float(obs.current_score - last_score))
                    last_score = obs.current_score
                    score = obs.current_score
                    
                except Exception as e:
                    error_msg = str(e)
                    action_str_log = f"error: {error_msg[:30]}" 
                    try:
                        res = await env.step(DataCleaningAction(command="submit", params={}))
                        obs = res.observation
                        obs_done = obs.done
                        step_reward = max(0.0, float(obs.current_score - last_score))
                        score = obs.current_score
                    except Exception:
                        obs_done = True
                
                rewards.append(step_reward)
                log_step(step=steps_taken, action=action_str_log, reward=step_reward, done=obs_done, error=error_msg)
                
                if obs_done:
                    break
                    
    except Exception as e:
        print(f"Environment error: {e}", file=sys.stderr)
    
    finally:
        # Clamping and final output (Emitted no matter what happens above)
        score = max(0.0, min(float(score), 1.0))
        success = score > 0.0 
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)

if __name__ == "__main__":
    asyncio.run(main())
