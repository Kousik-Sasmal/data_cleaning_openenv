import copy
from typing import List, Dict, Any

class DataCleaningTask:
    def __init__(self, name: str, description: str, initial_data: List[Dict[str, Any]], target_data: List[Dict[str, Any]]):
        self.name = name
        self.description = description
        self.initial_data = copy.deepcopy(initial_data)
        self.target_data = copy.deepcopy(target_data)

    def grader(self, current_data: List[Dict[str, Any]]) -> float:
        """
        Grader that computes a similarity score (0.0 to 1.0) between the submitted dataset and the target dataset.
        """
        if not current_data:
            return 0.0
        
        # Calculate how many rows match exactly
        match_count = 0
        target_copy = list(self.target_data)
        for current_row in current_data:
            if current_row in target_copy:
                match_count += 1
                target_copy.remove(current_row)
                
        # Calculate precision and recall
        precision = match_count / len(current_data) if len(current_data) > 0 else 0.0
        recall = match_count / len(self.target_data) if len(self.target_data) > 0 else 0.0
        
        # F1 score approximation capped to 1.0
        if precision + recall == 0:
            return 0.01
        score = 2 * (precision * recall) / (precision + recall)
        return max(0.01, min(0.99, float(score)))


# Task 1: Easy - Drop exact duplicates
TASK_1_INITIAL = [
    {"id": 1, "name": "Alice"},
    {"id": 1, "name": "Alice"}, # Duplicate
    {"id": 2, "name": "Bob"}
]
TASK_1_TARGET = [
    {"id": 1, "name": "Alice"},
    {"id": 2, "name": "Bob"}
]
task_easy = DataCleaningTask(
    name="Easy", 
    description="Remove exact duplicate rows from the dataset.",
    initial_data=TASK_1_INITIAL,
    target_data=TASK_1_TARGET
)

# Task 2: Medium - Fill NA and Drop Duplicates
TASK_2_INITIAL = [
    {"id": 1, "name": "Charlie", "email": None},
    {"id": 2, "name": "Dave", "email": "dave@example.com"},
    {"id": 2, "name": "Dave", "email": "dave@example.com"} # Duplicate
]
TASK_2_TARGET = [
    {"id": 1, "name": "Charlie", "email": "unknown"},
    {"id": 2, "name": "Dave", "email": "dave@example.com"}
]
task_medium = DataCleaningTask(
    name="Medium",
    description="Impute missing 'email' variables with 'unknown' and drop exact duplicate rows.",
    initial_data=TASK_2_INITIAL,
    target_data=TASK_2_TARGET
)

# Task 3: Hard - Format date, fill NA, filter
TASK_3_INITIAL = [
    {"id": 1, "date": "12/31/2023", "status": "active", "score": None},
    {"id": 2, "date": "01/15/2024", "status": "inactive", "score": 85},
    {"id": 3, "date": "02/20/2024", "status": "active", "score": 90}
]
TASK_3_TARGET = [
    {"id": 1, "date": "2023-12-31", "status": "active", "score": 0},
    {"id": 3, "date": "2024-02-20", "status": "active", "score": 90}
]
task_hard = DataCleaningTask(
    name="Hard",
    description="Format 'date' from MM/DD/YYYY to YYYY-MM-DD format. Fill missing 'score' with 0. Filter to keep only rows where 'status' is 'active'.",
    initial_data=TASK_3_INITIAL,
    target_data=TASK_3_TARGET
)

# Exposed globally for random selection or sequential iteration
TASKS = [task_easy, task_medium, task_hard]
