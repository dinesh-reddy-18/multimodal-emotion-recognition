import urllib.request
import urllib.parse
import json

sentences = [
    "I am so sad",
    "I feel extremely depressed and unhappy",
    "I am very happy today",
    "This is the best day of my life",
    "I am so angry right now",
    "I am terrified",
    "I feel scared",
    "I am surprised",
    "I feel calm and neutral",
    "I hate this situation",
    "I am crying and feeling miserable"
]

url = "http://127.0.0.1:8000/api/predict"

print("Running Text Prediction Tests...\n")
for idx, text in enumerate(sentences):
    data = urllib.parse.urlencode({"text": text}).encode("utf-8")
    req = urllib.request.Request(url, data=data, method="POST")
    try:
        with urllib.request.urlopen(req) as response:
            res = json.loads(response.read().decode("utf-8"))
            pred = res["final_prediction"]["label"]
            conf = res["final_prediction"]["confidence"]
            text_pred = res["modalities"]["text"]["label"]
            text_conf = res["modalities"]["text"]["confidence"]
            print(f"{idx+1}. Text: '{text}'")
            print(f"   Modality (Text) Prediction: {text_pred} ({text_conf*100:.1f}%)")
            print(f"   Final Prediction (Fused):   {pred} ({conf*100:.1f}%)")
            print("-" * 50)
    except Exception as e:
        print(f"ERROR for '{text}': {e}")
