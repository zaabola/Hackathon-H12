import csv
from collections import defaultdict

input_path = r"c:\Users\zaabola\Desktop\PPE_Project\datasets\crop_recommandation\Crop_recommendation.csv"
output_path = r"c:\Users\zaabola\Desktop\PPE_Project\datasets\gabes_crop_recommendation.csv"

# 1. Read the original data and calculate means
crop_data = defaultdict(lambda: {'temperature': 0, 'humidity': 0, 'ph': 0, 'rainfall': 0, 'count': 0})

with open(input_path, mode='r', encoding='utf-8') as infile:
    reader = csv.DictReader(infile)
    for row in reader:
        crop = row['label']
        crop_data[crop]['temperature'] += float(row['temperature'])
        crop_data[crop]['humidity'] += float(row['humidity'])
        crop_data[crop]['ph'] += float(row['ph'])
        crop_data[crop]['rainfall'] += float(row['rainfall'])
        crop_data[crop]['count'] += 1

# 2. Estimate water and O2
def estimate_water(crop):
    water_map = {
        'rice': 8.5, 'maize': 5.5, 'chickpea': 3.5, 'kidneybeans': 4.0, 'pigeonpeas': 4.0, 
        'mothbeans': 3.0, 'mungbean': 3.5, 'blackgram': 3.5, 'lentil': 3.5, 'pomegranate': 4.2, 
        'banana': 7.0, 'mango': 5.5, 'grapes': 4.5, 'watermelon': 6.5, 'melon': 6.0, 
        'apple': 5.0, 'orange': 5.0, 'papaya': 5.5, 'coconut': 6.5, 'cotton': 5.5, 
        'jute': 7.0, 'coffee': 5.0
    }
    return water_map.get(crop, 4.5)

def estimate_o2(crop):
    if crop in ['banana', 'mango', 'apple', 'orange', 'papaya', 'coconut', 'coffee']:
        return 'medium'
    return 'low'

final_data = []

# Process averages
for crop, totals in crop_data.items():
    count = totals['count']
    future_temp = (totals['temperature'] / count) + 2.0  # +2C for Gabes
    humidity = totals['humidity'] / count
    ph = totals['ph'] / count
    rainfall = totals['rainfall'] / count
    
    final_data.append({
        'Crop': crop,
        'Future_Optimal_Temp_C': round(future_temp, 2),
        'humidity': round(humidity, 2),
        'rainfall': round(rainfall, 2),
        'ph': round(ph, 2),
        'Water_Need_L_per_m2_day': estimate_water(crop),
        'O2_Production_Level': estimate_o2(crop)
    })

# 3. Add new crops
new_crops = [
    {'Crop': 'olive', 'Future_Optimal_Temp_C': 30.5, 'humidity': 38.0, 'rainfall': 200.0, 'ph': 7.5, 'Water_Need_L_per_m2_day': 3.5, 'O2_Production_Level': 'medium'},
    {'Crop': 'pistachio', 'Future_Optimal_Temp_C': 32.0, 'humidity': 35.0, 'rainfall': 180.0, 'ph': 7.8, 'Water_Need_L_per_m2_day': 2.8, 'O2_Production_Level': 'medium'},
    {'Crop': 'paulownia', 'Future_Optimal_Temp_C': 28.5, 'humidity': 55.0, 'rainfall': 500.0, 'ph': 6.5, 'Water_Need_L_per_m2_day': 9.0, 'O2_Production_Level': 'very_high'},
    {'Crop': 'moringa', 'Future_Optimal_Temp_C': 33.0, 'humidity': 35.0, 'rainfall': 200.0, 'ph': 6.8, 'Water_Need_L_per_m2_day': 3.8, 'O2_Production_Level': 'high'},
    {'Crop': 'neem', 'Future_Optimal_Temp_C': 34.0, 'humidity': 40.0, 'rainfall': 250.0, 'ph': 7.2, 'Water_Need_L_per_m2_day': 4.2, 'O2_Production_Level': 'high'},
    {'Crop': 'aloe vera', 'Future_Optimal_Temp_C': 35.0, 'humidity': 25.0, 'rainfall': 100.0, 'ph': 7.5, 'Water_Need_L_per_m2_day': 1.5, 'O2_Production_Level': 'medium'},
    {'Crop': 'date palm', 'Future_Optimal_Temp_C': 34.5, 'humidity': 30.0, 'rainfall': 150.0, 'ph': 8.0, 'Water_Need_L_per_m2_day': 5.5, 'O2_Production_Level': 'medium'},
    {'Crop': 'alfalfa', 'Future_Optimal_Temp_C': 29.5, 'humidity': 40.0, 'rainfall': 350.0, 'ph': 7.2, 'Water_Need_L_per_m2_day': 8.0, 'O2_Production_Level': 'medium'}
]

final_data.extend(new_crops)

# 4. Write to CSV
fieldnames = ['Crop', 'Future_Optimal_Temp_C', 'humidity', 'rainfall', 'ph', 'Water_Need_L_per_m2_day', 'O2_Production_Level']

with open(output_path, mode='w', newline='', encoding='utf-8') as outfile:
    writer = csv.DictWriter(outfile, fieldnames=fieldnames)
    writer.writeheader()
    for row in final_data:
        writer.writerow(row)

print(f"Generated successfully: {output_path}")
