curl \
-H "Content-Type: application/json" \
-d '{"TAC_Reading": 0.05, "energy": 0.2, "magnitude": 0.3, "time": 1735737600000, "x": 0.12, "y": 0.98, "z": -0.05}' \
-X POST http://127.0.0.1:5001/predict