"""
app.py
------
The dash app of the project
"""

import pandas as pd

# import model to generate project data
import model

(
    array_result,
    opt_result,
    A_p,
    A_g,
    nodes_p,
    nodes_g,
) = model.test_network_model()


nodemaster_p = pd.concat((opt_result[0]["nse_power"], nodes_p), axis=1)
nodemaster_p["relative_service"] = (
    1 - nodemaster_p["non_served_energy"] / nodemaster_p["load"]
)
nodemaster_p["index"] = nodemaster_p.index
print(nodemaster_p)
