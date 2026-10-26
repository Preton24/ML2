import pandas as pd
from itertools import product

POS = {"y", "yes", "true", "1", "positive"}
NEG = {"n", "no", "false", "0", "negative"}

def positive(label):
    return str(label).strip().lower() in POS

def consistent(h, x):
    return all(a == "?" or a == b for a, b in zip(h, x))

def general(h1, h2):
    return all(a == "?" or a == b for a, b in zip(h1, h2))

# Find-S
def find_s(df):
    h = None
    for _, row in df.iterrows():
        x = list(row[:-1].astype(str).str.strip())
        if positive(row.iloc[-1]):
            if h is None:
                h = x
            else:
                h = [a if a == b else "?" for a, b in zip(h, x)]
    return h

# Candidate Elimination
def candidate_elimination(df):
    n = df.shape[1] - 1
    domains = [df.iloc[:, i].astype(str).str.strip().unique()
               for i in range(n)]
    S, G = [["0"] * n], [["?"] * n]

    for _, row in df.iterrows():
        x = list(row[:-1].astype(str).str.strip())
        label = str(row.iloc[-1]).strip()

        if positive(label):
            G = [g for g in G if consistent(g, x)]
            S = [x.copy() if s == ["0"] * n else
                 [a if a == b else "?" for a, b in zip(s, x)]
                 for s in S]
        elif label.lower() in NEG:
            S = [s for s in S if not consistent(s, x)]
            new_G = []
            for g in G:
                if consistent(g, x):
                    for i in range(n):
                        if g[i] == "?":
                            for v in domains[i]:
                                if v != x[i]:
                                    h = g.copy()
                                    h[i] = v
                                    if any(general(h, s) for s in S):
                                        new_G.append(h)
                else:
                    new_G.append(g)
            G = new_G

        S = [s for s in S if any(general(g, s) for g in G)]
        G = [g for g in G if any(general(g, s) for s in S)]

    return S, G

# Read dataset
df = pd.read_csv("C:/Users/gagan/Downloads/Dataset_1.csv", dtype=str)

print("Dataset:\n", df.head())
print("\n\nFind-S:", find_s(df))

S, G = candidate_elimination(df)
print("Specific boundary S:", S)
print("General boundary G:", G)