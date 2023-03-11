import pandas as pd
df = pd.read_csv("./imaterialist/metrics.csv")
print(df["IoU"].mean())