import csv
import json

# Girdi CSV dosya adı
csv_files = ['trainingData.csv','validationData.csv']



for csv_file in csv_files:

    # Boş bir sözlük oluşturuyoruz
    data = {}
    # Çıktı JSON dosya adı
    json_file = csv_file.replace('.csv', '-GT.json')
    # CSV dosyasını aç
    with open(csv_file, mode='r', encoding='utf-8') as file:
        reader = csv.DictReader(file)
        
        # Her satırı sırayla işle
        for i, row in enumerate(reader, start=1):
            # Yalnızca istenen sütunları al
            data[str(i-1)] = {
                "FLOOR": row.get("FLOOR"),
                "BUILDINGID": row.get("BUILDINGID"),
                "SPACEID": row.get("SPACEID")
            }

    # JSON dosyasına yaz
    with open(json_file, mode='w', encoding='utf-8') as file:
        json.dump(data, file, ensure_ascii=False, indent=4)

    print(f"✅ JSON dosyası '{json_file}' başarıyla oluşturuldu!")