import json
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Any
from collections import defaultdict
import matplotlib.pyplot as plt
from src.config.path_config import PathConfig

class LogAnalyzer:
    def __init__(self):
        self.path_config = PathConfig()
        self.chat_logs_dir = self.path_config.CHAT_LOGS
        self.process_logs_dir = self.path_config.PROCESS_LOGS

    def analyze_chat_history(self, days: int = 7) -> Dict[str, Any]:
        """Analyze chat history for the specified number of days"""
        chat_data = []
        
        # Load all chat logs
        for log_file in self.chat_logs_dir.glob("*.json"):
            with open(log_file, 'r', encoding='utf-8') as f:
                chat_data.extend(json.load(f))

        # Convert to DataFrame
        df = pd.DataFrame(chat_data)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # Filter for specified days
        cutoff_date = datetime.now() - timedelta(days=days)
        df = df[df['timestamp'] > cutoff_date]

        analysis = {
            "total_interactions": len(df),
            "average_rating": df['rating'].mean(),
            "rating_distribution": df['rating'].value_counts().to_dict(),
            "interactions_by_day": df.groupby(df['timestamp'].dt.date).size().to_dict(),
            "common_topics": self._extract_common_topics(df['user_input'].tolist()),
        }

        return analysis

    def generate_report(self, days: int = 7) -> str:
        """Generate a detailed report of system performance"""
        analysis = self.analyze_chat_history(days)
        
        report = f"""
AI Support System Analysis Report
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Period: Last {days} days

Performance Metrics:
------------------
Total Interactions: {analysis['total_interactions']}
Average Rating: {analysis['average_rating']:.2f}

Rating Distribution:
------------------
"""
        for rating, count in analysis['rating_distribution'].items():
            report += f"Rating {rating}: {count} interactions\n"

        report += "\nDaily Interaction Counts:\n"
        for date, count in analysis['interactions_by_day'].items():
            report += f"{date}: {count} interactions\n"

        report += "\nCommon Topics:\n"
        for topic, count in analysis['common_topics'].items():
            report += f"{topic}: {count} occurrences\n"

        return report

    def plot_ratings_distribution(self, save_path: str = None):
        """Generate a plot of rating distribution"""
        analysis = self.analyze_chat_history()
        
        plt.figure(figsize=(10, 6))
        ratings = list(analysis['rating_distribution'].keys())
        counts = list(analysis['rating_distribution'].values())
        
        plt.bar(ratings, counts)
        plt.title('Distribution of Response Ratings')
        plt.xlabel('Rating')
        plt.ylabel('Number of Responses')
        
        if save_path:
            plt.savefig(save_path)
        else:
            plt.show()

    def _extract_common_topics(self, user_inputs: List[str], top_n: int = 10) -> Dict[str, int]:
        """Extract common topics from user inputs"""
        # Simple keyword extraction (can be enhanced with NLP)
        keywords = defaultdict(int)
        common_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for'}
        
        for input_text in user_inputs:
            words = input_text.lower().split()
            for word in words:
                if word not in common_words and len(word) > 3:
                    keywords[word] += 1
        
        # Return top N topics
        return dict(sorted(keywords.items(), key=lambda x: x[1], reverse=True)[:top_n])

    def export_to_excel(self, output_path: str):
        """Export analysis to Excel"""
        analysis = self.analyze_chat_history()
        
        with pd.ExcelWriter(output_path) as writer:
            # Overall metrics
            pd.DataFrame([{
                'Total Interactions': analysis['total_interactions'],
                'Average Rating': analysis['average_rating']
            }]).to_excel(writer, sheet_name='Overall Metrics', index=False)
            
            # Rating distribution
            pd.DataFrame(analysis['rating_distribution'].items(), 
                        columns=['Rating', 'Count']).to_excel(writer, sheet_name='Rating Distribution', index=False)
            
            # Daily interactions
            pd.DataFrame(analysis['interactions_by_day'].items(),
                        columns=['Date', 'Interactions']).to_excel(writer, sheet_name='Daily Interactions', index=False)
            
            # Common topics
            pd.DataFrame(analysis['common_topics'].items(),
                        columns=['Topic', 'Occurrences']).to_excel(writer, sheet_name='Common Topics', index=False) 