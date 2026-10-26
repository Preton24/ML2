import pandas as pd
import math

df = pd.read_csv("C:/Users/gagan/Downloads/CN2 FOIL.csv", dtype=str).apply(lambda c: c.str.strip())
target = df.columns[-1]
attrs = list(df.columns[:-1])
BEAM = 3

def entropy(data):
    e = 0
    for p in data[target].value_counts(normalize=True):
        e -= p * math.log2(p)
    return e

def score(rule):
    d = df.copy()
    for a, v in rule.items():
        d = d[d[a] == v]
    return -entropy(d) if len(d) else -999

beam = [{}]
best = {}

while beam:
    rules = []
    for r in beam:
        for a in attrs:
            if a not in r:
                for v in df[a].unique():
                    new = r.copy()
                    new[a] = v
                    rules.append(new)

    if not rules:
        break

    rules.sort(key=score, reverse=True)
    if score(rules[0]) > score(best):
        best = rules[0]

    beam = rules[:BEAM]

    d = df.copy()
    for a, v in best.items():
        d = d[d[a] == v]
    if len(d) and entropy(d) == 0:
        break

print("Best Rule:", best)

d = df.copy()
for a, v in best.items():
    d = d[d[a] == v]

if best and len(d):
    print("IF", best, "THEN", target, "=", d[target].mode()[0])
else:
    print("No rule found")