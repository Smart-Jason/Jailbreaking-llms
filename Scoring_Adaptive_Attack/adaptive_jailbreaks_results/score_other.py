import os
import json
import re
from dotenv import load_dotenv
from judges import get_judge_prompt
from together import Together

load_dotenv() 

together = Together(api_key=os.environ["TOGETHER_API_KEY"])

input_file = "exps_mistral-7b_simplified_template.json"
output_file = "log/gemma3nE4B_judged.json"
model = "google/gemma-3n-E4B-it"
os.makedirs("log", exist_ok=True)

with open(input_file, "r", encoding="utf-8") as f:
    data = json.load(f)

results = []
total = len(data["jailbreaks"])

for idx, jb in enumerate(data["jailbreaks"]):
    goal = jb["goal"]
    response = jb["response"]

    print(f"[{idx+1}/{total}] Judging response for goal: {goal[:60]}...")

    try:
        judge_prompt = get_judge_prompt(goal, response)
        judge_reply = together.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": judge_prompt}],
            max_tokens=256,
            temperature=0.0
        )

        raw_output = judge_reply.choices[0].message.content.strip()
        raw_output = raw_output.replace("```json", "").replace("```", "").strip()
        match = re.search(r'\{.*\}', raw_output, re.DOTALL)

        if match:
            score_json = json.loads(match.group(0))
        else:
            raise ValueError("No valid JSON object found in output.")

        print(f"    ✅ Judged. H_score: {score_json.get('H_score')}, R_score: {score_json.get('R_score')}")

    except Exception as e:
        score_json = {
            "H_score": -1,
            "R_score": -1,
            "H_rationale": f"Error: {str(e)}",
            "R_rationale": f"Error: {str(e)}"
        }
        print(f"    ❌ Failed: {e}")

    results.append({
        "goal": goal,
        "response": response,
        "score": score_json
    })

with open(output_file, "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2)

print(f"\n✅ All finished. Results saved to {output_file}")
