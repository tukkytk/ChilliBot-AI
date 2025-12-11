start cmd /k python -m uvicorn main:app --host 0.0.0.0 --port 5000
timeout 3
start cmd /k ngrok http 5000
