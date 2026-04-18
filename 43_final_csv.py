import pandas as pd

# -----------------------------
# Load datasets
# -----------------------------
soil_df = pd.read_csv(r"c:\Users\zaabola\Desktop\PPE_Project\datasets\gabescsv\gabes_soil_dataset.csv")
crop_df = pd.read_csv(r"c:\Users\zaabola\Desktop\PPE_Project\datasets\gabes_crop_recommendation.csv")

# -----------------------------
# Normalize column names (safety)
# -----------------------------
soil_df.columns = soil_df.columns.str.strip().str.lower()
crop_df.columns = crop_df.columns.str.strip().str.lower()

# -----------------------------
# RULE FUNCTION
# -----------------------------
def is_crop_valid(soil_type, salinity_level, crop_row):

    salinity = str(salinity_level).lower()
    soil = str(soil_type).lower()
    crop = crop_row["crop"].lower()
    water = crop_row["water_need_l_per_m2_day"]
    suitability = crop_row["suitability_for_gabes"]
    salinity_tol = crop_row.get("salinity_tolerance", "medium")

    # -------------------------
    # 1. SALTY SOIL (HALOMORPHIC)
    # -------------------------
    if soil == "halomorphic":
        return salinity_tol in ["high", "very_high"]

    # -------------------------
    # 2. FERTILE SOIL
    # -------------------------
    if soil == "alluvial_fertile":
        return suitability in ["high", "medium"]

    # -------------------------
    # 3. DESERT / SANDY
    # -------------------------
    if soil in ["arid_sandy", "desert"]:
        return water <= 4 and salinity_tol in ["medium", "high", "very_high"]

    # -------------------------
    # 4. MIXED / OTHER
    # -------------------------
    return suitability == "high"


# -----------------------------
# GENERATE FINAL DATASET
# -----------------------------
final_rows = []

for _, soil in soil_df.iterrows():
    for _, crop in crop_df.iterrows():

        if is_crop_valid(soil["soil_type"], soil["salinity_level"], crop):

            final_rows.append({
                "zone": soil["zone"],
                "soil_type": soil["soil_type"],
                "salinity": soil["salinity_level"],
                "crop": crop["crop"],
                "water_need_l_per_m2_day": crop["water_need_l_per_m2_day"],
                "suitability": crop["suitability_for_gabes"]
            })

# -----------------------------
# SAVE RESULT
# -----------------------------
final_df = pd.DataFrame(final_rows)

final_df.to_csv("gabes_crop_mapping.csv", index=False)

print("Final CSV created: gabes_crop_mapping.csv")
print(final_df.head())