curl \
-H "Content-Type: application/json" \
-d '{"time": 1735737600000, "x": 0.12, "y": 0.98, "z": -0.05, "TAC_Reading": 0.9, "energy": 0.3, "magnitude": 0.7}' \
-X POST http://127.0.0.1:5001/predict