import subprocess
import time
import requests
import json

def main():
    print("Starting FastAPI backend uvicorn server...")
    proc = subprocess.Popen(
        ["venv\\Scripts\\python.exe", "-m", "uvicorn", "backend.app.main:app", "--port", "8081", "--host", "127.0.0.1"]
    )
    
    # Wait for startup (ONNX model loading on CPU takes some time)
    time.sleep(25)
    
    try:
        print("Sending prediction request for text modality...")
        url = "http://127.0.0.1:8081/api/predict"
        data = {"text": "I was walking home alone when I noticed the same car moving slowly behind me for several streets."}
        res = requests.post(url, data=data)
        
        print("Status code:", res.status_code)
        if res.status_code == 200:
            print("Response JSON:")
            print(json.dumps(res.json(), indent=2))
        else:
            print("Error response:", res.text)
    except Exception as e:
        print("Test failed with exception:", e)
    finally:
        print("Terminating backend server...")
        proc.terminate()
        proc.wait()
        print("Server stopped.")

if __name__ == "__main__":
    main()
