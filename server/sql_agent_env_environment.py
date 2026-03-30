# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Sql Agent Env Environment Implementation.

A simple test environment that echoes back messages sent to it.
Perfect for testing HTTP server infrastructure.
"""

from uuid import uuid4

import copy
from datetime import datetime
from uuid import uuid4
from typing import Any

from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import State

try:
    from ..models import SqlAgentAction, SqlAgentObservation
    from .tasks import TASKS
except ImportError:
    from models import SqlAgentAction, SqlAgentObservation
    from server.tasks import TASKS


class SqlAgentEnvironment(Environment):
    """
    Data Cleaning Environment where the agent processes messy datasets.
    """

    SUPPORTS_CONCURRENT_SESSIONS: bool = True

    def __init__(self):
        """Initialize the data cleaning environment."""
        self._state = State(episode_id=str(uuid4()), step_count=0)
        self._reset_count = 0
        self.current_task = None
        self._dataset = []
        self._last_score = 0.0

    def reset(self, seed=None, task_id=None, **kwargs) -> SqlAgentObservation:
        self._state = State(episode_id=str(uuid4()), step_count=0)
        self._reset_count += 1
        
        # Pick task
        if task_id is not None and 0 <= task_id < len(TASKS):
            self.current_task = TASKS[task_id]
        else:
            self.current_task = TASKS[self._reset_count % len(TASKS)]
            
        self._dataset = copy.deepcopy(self.current_task.initial_data)
        self._last_score = self.current_task.grader(self._dataset)
        
        return self._build_obs(f"Started Task: {self.current_task.description}", done=False, reward=0.0)

    def _build_obs(self, message: str, done: bool, reward: float) -> SqlAgentObservation:
        preview = str(self._dataset[:5]) if self._dataset else "Empty dataset"
        schema = str(list(self._dataset[0].keys())) if self._dataset else "None"
        
        score = self.current_task.grader(self._dataset) if self.current_task else 0.0
        
        return SqlAgentObservation(
            dataset_preview=preview,
            schema_info=schema,
            message=message,
            current_score=score,
            done=done,
            reward=reward
        )

    def step(self, action: SqlAgentAction, timeout_s=None, **kwargs) -> SqlAgentObservation:
        self._state.step_count += 1
        cmd = action.command
        params = action.params
        msg = ""
        done = False
        
        try:
            if cmd == "drop_duplicates":
                seen = []
                new_data = []
                for row in self._dataset:
                    if row not in seen:
                        seen.append(row)
                        new_data.append(row)
                self._dataset = new_data
                msg = "Dropped duplicates."
                
            elif cmd == "fill_na":
                col = params.get("column")
                val = params.get("value")
                for row in self._dataset:
                    if row.get(col) is None:
                        row[col] = val
                msg = f"Filled NA in {col} with {val}."
                
            elif cmd == "format_date":
                col = params.get("column")
                for row in self._dataset:
                    if row.get(col):
                        try:
                            d = datetime.strptime(row[col], "%m/%d/%Y")
                            row[col] = d.strftime("%Y-%m-%d")
                        except ValueError:
                            pass
                msg = f"Formatted date in {col}."
                
            elif cmd == "filter":
                col = params.get("column")
                val = params.get("value")
                self._dataset = [r for r in self._dataset if str(r.get(col)) == str(val)]
                msg = f"Filtered {col} == {val}."
                
            elif cmd == "submit":
                msg = "Submitted solution."
                done = True
                
            else:
                msg = f"Unknown command: {cmd}"
                
        except Exception as e:
            msg = f"Error processing command: {str(e)}"

        # Compute partial/final reward based on progress
        score = self.current_task.grader(self._dataset) if self.current_task else 0.0
        
        if done:
            reward = score  # Final reward is the grader score
        else:
            reward = max(0.0, score - self._last_score) # Sparse positive reward for progress
            
        self._last_score = score
        
        return self._build_obs(msg, done, reward)

    @property
    def state(self) -> State:
        return self._state
