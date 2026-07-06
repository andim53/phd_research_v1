import csv
import json



input_csv = "d4_5.csv"
output_json = "output.json"

structures = []

with open(input_csv, newline="") as f:
    reader = csv.reader(f)
    for row in reader:
        # CSV形式:
        # 元素, 組成比, δ, 番号
        elements_str = row[0]          # 例: V-Cr-Mn-Zn-Pt
        ratio_str = row[1]             # 例: 7-7-7-7-4

        elems = elements_str.split("-")
        ratios = list(map(int, ratio_str.split("-")))

        element_list = []
        for e, n in zip(elems, ratios):
            element_list += [e] * n

        structures.append({
            "name": elements_str,
            "elements": element_list
        })

with open(output_json, "w") as f:
    json.dump({"structures": structures}, f, indent=2)

with open("output.json") as f:
    data = json.load(f)

for structure in data["structures"]:
    name = structure["name"]
    elements = structure["elements"]

    print("name:", name)
    print("elements:", elements)
    print()

print("done")