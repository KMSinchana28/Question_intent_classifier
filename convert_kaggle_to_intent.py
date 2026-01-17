import pandas as pd

# Load Kaggle TREC dataset
df = pd.read_csv("data/train.csv")

def map_intent(row):
    q = row["text"].lower().strip()
    label = row["label-coarse"]

    # Pattern-based rules (highest priority)
    if q.startswith("how"):
        return "HowTo"
    if q.startswith("why"):
        return "Reason"
    if q.startswith(("is", "are", "can", "does", "do", "did")):
        return "YesNo"
    if "compare" in q or "difference" in q:
        return "Comparison"
    if q.startswith(("who", "when", "where")):
        return "Fact"

    # Label-based fallback
    if label in [0, 2]:   # ABBR, DESC
        return "Definition"
    else:
        return "Fact"

# Apply mapping
df["intent"] = df.apply(map_intent, axis=1)

# Keep required columns only
final_df = df[["text", "intent"]]
final_df.columns = ["question", "intent"]

# Save final dataset
final_df.to_csv("data/questions.csv", index=False)

print("✅ NEW questions.csv created from Kaggle dataset")
print("\nIntent distribution:")
print(final_df["intent"].value_counts())
