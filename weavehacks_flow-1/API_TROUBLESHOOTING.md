# API Troubleshooting Guide

## Common API Errors and Solutions

### Error 400: Bad Request (Experiment Creation)

**Problem**: When creating an experiment through Streamlit, you get "API Error: 400"

**Cause**: The experiment ID is being sent incorrectly to the API

**Solution**: The API has been fixed to accept experiment_id as a query parameter:
```python
# Correct way:
response = requests.post(f"{API_BASE_URL}/experiments?experiment_id={experiment_id}")
```

### Error 500: Internal Server Error (Voice Entry)

**Problem**: Manual data entry via voice gives "API Error: 500"

**Possible Causes**:
1. Missing timestamp field in DataEntry
2. Backend expecting different data format
3. Chemistry calculation endpoints have wrong names

**Solutions Applied**:
1. Fixed endpoint names:
   - `sulfur-amount` → `sulfur_amount`
   - `nabh4-amount` → `nabh4_amount`
   - `percent-yield` → `percent_yield`

2. Fixed calculation parameters:
   - `final_mass` → `actual_yield` for percent yield calculation

3. Fixed result field names:
   - `mass_sulfur_needed_g` → `mass_sulfur_g`
   - `mass_nabh4_needed_g` → `mass_nabh4_g`

## Testing the API

### 1. Start the Backend
```bash
# Option 1: Using the helper script
python3 start_backend.py

# Option 2: Manual start
cd backend
python3 -m uvicorn main:app --reload
```

### 2. Test API Endpoints
```bash
# Run the test script to verify all endpoints
python3 test_api.py
```

### 3. Check Backend Logs
The backend will show detailed logs for each request. Look for:
- Request method and path
- Request parameters
- Any Python exceptions

## Manual API Testing

### Create Experiment
```bash
curl -X POST "http://localhost:8000/experiments?experiment_id=test_001"
```

### Record Data
```bash
curl -X POST "http://localhost:8000/data" \
  -H "Content-Type: application/json" \
  -d '{
    "experiment_id": "test_001",
    "data_type": "mass",
    "compound": "HAuCl₄·3H₂O",
    "value": 0.1576,
    "units": "g"
  }'
```

### Calculate Chemistry
```bash
# Sulfur amount
curl -X POST "http://localhost:8000/calculations/sulfur_amount?experiment_id=test_001&gold_mass=0.1576"

# NaBH4 amount
curl -X POST "http://localhost:8000/calculations/nabh4_amount?experiment_id=test_001&gold_mass=0.1576"
```

## Common Issues

### 1. Backend Not Running
- **Symptom**: "Connection Error" in Streamlit
- **Fix**: Start the backend with `python3 start_backend.py`

### 2. Missing Dependencies
- **Symptom**: ImportError when starting backend
- **Fix**: `pip3 install --user fastapi uvicorn pydantic`

### 3. Port Already in Use
- **Symptom**: "Address already in use" error
- **Fix**: Kill the process using port 8000 or change the port

### 4. CORS Issues
- **Symptom**: Browser console shows CORS errors
- **Fix**: The backend already has CORS middleware configured for all origins

## Debugging Tips

1. **Check Backend Console**: Most errors will show detailed stack traces in the backend terminal

2. **Use test_api.py**: This script tests all endpoints and shows exactly what data is being sent

3. **Check Data Types**: Ensure timestamps are ISO format strings, not datetime objects

4. **Verify Experiment Exists**: Many endpoints return 404 if the experiment_id doesn't exist

5. **Check Request Format**: The backend expects:
   - Query parameters for: experiment_id, gold_mass, etc.
   - JSON body for: DataEntry, SafetyAlert, etc.