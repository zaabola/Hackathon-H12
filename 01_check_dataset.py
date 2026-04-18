import os
import xml.etree.ElementTree as ET
from collections import Counter

helmet_xml_path = "datasets/helmet_dataset/Annotations"

xml_files = [f for f in os.listdir(helmet_xml_path) if f.endswith(".xml")]

class_counter = Counter()
problem_files = []

for xml_file in xml_files:
    xml_path = os.path.join(helmet_xml_path, xml_file)
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()

        for obj in root.findall("object"):
            name = obj.find("name").text.strip()
            class_counter[name] += 1

    except Exception as e:
        problem_files.append((xml_file, str(e)))

print("✅ ALL CLASSES FOUND:")
print(class_counter)

print("\n📊 Class counts:")
for cls, count in class_counter.items():
    print(f"{cls}: {count}")

print("\n📁 Total XML files:", len(xml_files))
print("⚠️ Problem files:", len(problem_files))

if problem_files[:10]:
    print("\nFirst problem files:")
    for pf in problem_files[:10]:
        print(pf)