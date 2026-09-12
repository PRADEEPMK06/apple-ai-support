import sys
import json
from pathlib import Path

# Add project root to path
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

from src.pipeline import run_pipeline

"""
demo.py

This script allows you to test the complete end-to-end AI support agent via the 
command line. It accepts a customer message as an argument and prints a formatted
summary of every stage in the pipeline.
"""

def main():
    if len(sys.argv) < 2:
        print("Usage: python scripts/demo.py \"<customer message>\"")
        print("Example: python scripts/demo.py \"My battery drains so fast after iOS update\"")
        sys.exit(1)

    customer_message = sys.argv[1]
    
    print("\n" + "="*60)
    print("🚀 RUNNING AI SUPPORT AGENT PIPELINE")
    print("="*60)
    print(f"Customer Message: \"{customer_message}\"\n")
    
    # The pipeline components have their own progress prints.
    result = run_pipeline(customer_message)
    
    print("\n" + "="*60)
    print("✅ PIPELINE COMPLETE. FINAL OUTPUT:")
    print("="*60)
    
    # Print the JSON beautifully
    print(json.dumps(result, indent=2))
    print("="*60 + "\n")

if __name__ == "__main__":
    main()
