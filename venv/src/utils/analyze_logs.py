import argparse
from log_analyzer import LogAnalyzer
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(description='Analyze AI Support System Logs')
    parser.add_argument('--days', type=int, default=7, help='Number of days to analyze')
    parser.add_argument('--export', type=str, help='Export analysis to Excel file')
    parser.add_argument('--plot', action='store_true', help='Generate rating distribution plot')
    
    args = parser.parse_args()
    
    analyzer = LogAnalyzer()
    
    # Print report
    print(analyzer.generate_report(args.days))
    
    # Generate plot if requested
    if args.plot:
        plot_path = Path('analysis_plots')
        plot_path.mkdir(exist_ok=True)
        analyzer.plot_ratings_distribution(str(plot_path / 'ratings_distribution.png'))
    
    # Export to Excel if requested
    if args.export:
        analyzer.export_to_excel(args.export)

if __name__ == '__main__':
    main() 