import sys
import pandas as pd
from datetime import date

day = int(sys.argv[1])
hours = sys.argv[2] if len(sys.argv) > 2 else ""
df = pd.read_csv("progress_tracker.csv", dtype=str).fillna("")
df.loc[df["Day"] == str(day), ["Status", "Date", "Hours"]] = ["Done", str(date.today()), hours]
df.to_csv("progress_tracker.csv", index=False)
print(df[["Day", "Topic", "Status"]].to_string(index=False))
