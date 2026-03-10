import requests

payload = {
    "owner_name": "VJAY",
    "rent_per_month": 500,
    "max_occupants": 5,
    "property_type": "FLAT",
    "floor": "2",
    "pet_friendly": True,
    "address": "VISAKHAPATNAM",
}

response = requests.post("http://127.0.0.1:8000/host-property", json=payload, timeout=90)
print(response.status_code)
print(response.text)
