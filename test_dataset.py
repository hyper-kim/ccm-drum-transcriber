import urllib.request
import json

# Check if we can reach MDB-Drums on Zenodo
try:
    req = urllib.request.Request("https://zenodo.org/api/records/3358742")
    with urllib.request.urlopen(req) as response:
        data = json.loads(response.read().decode())
        print("MDB-Drums Zenodo Record found!")
        for f in data.get('files', []):
            print(f['key'], f['size'] / 1024 / 1024, "MB")
except Exception as e:
    print("Zenodo error:", e)
