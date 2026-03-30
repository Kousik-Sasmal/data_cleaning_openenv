# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Data models for the Sql Agent Env Environment.

The sql_agent_env environment is a simple test environment that echoes back messages.
"""

from openenv.core.env_server.types import Action, Observation
from pydantic import Field


from typing import Dict, Any, Optional
from openenv.core.env_server.types import Action, Observation
from pydantic import Field


class SqlAgentAction(Action):
    """Action for the Data Cleaning environment."""

    command: str = Field(
        ..., 
        description="Data manipulation command: 'drop_duplicates', 'fill_na', 'format_date', 'filter', or 'submit'."
    )
    params: Dict[str, Any] = Field(
        default_factory=dict, 
        description="Parameters for the command (e.g., {'column': 'email', 'value': 'unknown'})."
    )


class SqlAgentObservation(Observation):
    """Observation from the Data Cleaning environment."""

    dataset_preview: str = Field(default="", description="String preview of the current dataset (first 5 rows).")
    schema_info: str = Field(default="", description="Information about available columns.")
    message: str = Field(default="", description="Feedback from the last executed action.")
    current_score: float = Field(default=0.0, description="Current grader score indicating progress.")
    done: bool = Field(default=False)
    reward: float = Field(default=0.0)
