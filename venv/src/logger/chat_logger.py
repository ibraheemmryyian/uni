import json
from datetime import datetime
from pathlib import Path

class ChatLogger:
    def __init__(self):
        self.log_dir = Path("venv/logs/chat_history")
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.current_file = self._create_log_file()

    def _create_log_file(self) -> Path:
        """Create a new log file with timestamp"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return self.log_dir / f"chat_history_{timestamp}.json"

    def log_interaction(self, user_input: str, ai_response: str, rating: int = None):
        """Log a single interaction with timestamp and rating"""
        interaction = {
            "timestamp": datetime.now().isoformat(),
            "user_input": user_input,
            "ai_response": ai_response,
            "rating": rating
        }

        # Read existing logs
        if self.current_file.exists():
            with open(self.current_file, 'r', encoding='utf-8') as f:
                logs = json.load(f)
        else:
            logs = []

        # Append new interaction
        logs.append(interaction)

        # Write updated logs
        with open(self.current_file, 'w', encoding='utf-8') as f:
            json.dump(logs, f, indent=2, ensure_ascii=False) 