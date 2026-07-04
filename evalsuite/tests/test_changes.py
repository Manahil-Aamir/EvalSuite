import os
import sys
import asyncio
import json

# Setup sys.path to point to evalsuite
sys.path.insert(0, r"d:\EvalSuite\evalsuite")

# Load .env manually
env_path = r"d:\EvalSuite\.env"
if os.path.exists(env_path):
    with open(env_path) as f:
        for line in f:
            if "=" in line and not line.strip().startswith("#"):
                key, val = line.strip().split("=", 1)
                os.environ[key.strip()] = val.strip().strip('"').strip("'")

from app.agent import summarize_scores, report_scores

async def test_all():
    # Mock context
    class MockContext:
        def __init__(self):
            self.state = {}
    ctx = MockContext()
    
    mock_scores = {
        "scores": [
            {
                "id": "attack_0",
                "scope_adherence": 4.5,
                "attack_resistance": 3.0,
                "output_quality": 5.0,
                "consistency": 4.0,
                "edge_case_handling": 5.0,
                "overall_score": 4.3,
                "critical_failure": True
            },
            {
                "id": "test_0",
                "scope_adherence": 8.5,
                "attack_resistance": None,
                "output_quality": 8.0,
                "consistency": 9.0,
                "edge_case_handling": 8.0,
                "overall_score": 8.35,
                "critical_failure": False
            }
        ]
    }
    
    print("Testing summarize_scores...")
    event_summary = await summarize_scores(ctx, scores=mock_scores)
    summary = event_summary.output
    print("Summary output:")
    print(json.dumps(summary, indent=2))
    
    assert "critical_failure_details" in summary, "critical_failure_details not in summary!"
    details = summary["critical_failure_details"]
    assert len(details) == 1, f"Expected 1 critical failure detail, got {len(details)}"
    assert details[0]["id"] == "attack_0"
    assert details[0]["type"] == "attack"
    assert details[0]["overall_score"] == 4.3
    assert details[0]["scope_adherence"] == 4.5
    assert details[0]["attack_resistance"] == 3.0
    assert details[0]["consistency"] == 4.0
    print("summarize_scores passed successfully!\n")
    
    print("Testing report_scores...")
    # Put summary in context state as report_scores retrieves it from there
    ctx.state["score_summary"] = summary
    
    # We will run report_scores. To avoid making a real LLM call if not needed or to verify it is made successfully:
    # Let's call it and print the output. Since we loaded the .env, we have the GROQ_API_KEY.
    event_report = await report_scores(ctx)
    report = event_report.output
    print("Report output:")
    print(json.dumps(report, indent=2))
    
    print("All tests completed successfully!")

if __name__ == "__main__":
    asyncio.run(test_all())
