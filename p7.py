import pandas as pd
from pgmpy.models import DiscreteBayesianNetwork
from pgmpy.parameter_estimator import DiscreteBayesianEstimator
from pgmpy.inference import VariableElimination

df = pd.read_csv("C:/Users/gagan/Downloads/student_data.csv")

model = DiscreteBayesianNetwork([
    ("Attendance", "Result"),
    ("Study", "Result")
])

estimator = DiscreteBayesianEstimator(
    prior_type="BDeu",
    equivalent_sample_size=5
)

model.fit(df, estimator=estimator)

infer = VariableElimination(model)

result = infer.query(
    variables=["Result"],
    evidence={
        "Attendance": "High",
        "Study": "High"
    }
)

print(result)

states = result.state_names["Result"]
pass_label = "Pass"  # Change this to match your dataset's exact label

if pass_label not in states:
    raise ValueError(
        f"Passing label {pass_label!r} not found. "
        f"Available states: {states}"
    )

pass_probability = result.values[states.index(pass_label)] * 100

print(f"Probability of passing: {pass_probability:.2f}%")