import json
from huggingface_hub import hf_hub_download

print("Downloading labeled.jsonl from HF...")
filepath = hf_hub_download(repo_id="AnishRacherla/LinguaFranca-Phase3", filename="data/2wikimultihopqa/labeled.jsonl", repo_type="dataset")

print("Checking for blank gold entities in labeled.jsonl...")
with open(filepath, encoding="utf-8") as f:
    hops = [h for l in f for h in json.loads(l).get("hops", [])]
blank = sum(1 for h in hops if not h.get("bridging_entity_gold", "").strip())
print(f"{blank}/{len(hops)} hops have a blank gold entity ({100*blank/len(hops):.1f}%)")
