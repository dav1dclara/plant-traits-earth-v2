import numpy as np
import pandas as pd
from scipy.special import inv_boxcox1p

# ----------------------------
# FILE PATHS
# ----------------------------
excel_file = "trait_map_values.xlsx"  # input Excel file
params_file = "power_transformer_params.csv"  # parameter file
output_file = "trait_map_original_units.xlsx"  # output file


# ----------------------------
# LOAD DATA
# ----------------------------
df = pd.read_excel(excel_file)
params = pd.read_csv(params_file).set_index("trait")


# ----------------------------
# BACK TRANSFORM FUNCTION
# ----------------------------
def back_transform(values, trait):
    """
    Convert standardized Yeo-Johnson values
    back to original trait units.
    """

    p = params.loc[trait]

    lmbda = p["yeo_johnson_lambda"]
    mean = p["standardize_mean"]
    scale = p["standardize_scale"]

    # Step 1: Un-standardize
    y = np.asarray(values, dtype=float) * scale + mean

    # Step 2: Inverse Yeo-Johnson
    x = np.where(
        y >= 0,
        inv_boxcox1p(y, lmbda),
        -inv_boxcox1p(-y, 2 - lmbda),
    )

    return x


# ----------------------------
# APPLY TRANSFORM
# ----------------------------
result = df.copy()

for trait in df.columns:
    if trait in params.index:
        result[trait] = back_transform(df[trait].values, trait)
    else:
        print(f"Skipping column {trait} (no parameters found)")


# ----------------------------
# SAVE OUTPUT
# ----------------------------
result.to_excel(output_file, index=False)

print("Back-transformation complete.")
print(f"Saved to: {output_file}")
