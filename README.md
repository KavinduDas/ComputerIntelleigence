# Academic Performance Predictor

A Flask web application that predicts exam scores based on student ID using a trained XGBoost machine learning model.

## Setup Instructions

1. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

2. **Prepare Data Files**
   - Ensure `train.csv` is in the `Dataset/` folder (or root directory)
   - Ensure `test.csv` is in the root directory (or `Dataset/` folder)

3. **Run the Application**
   ```bash
   python app.py
   ```

4. **Access the Web Interface**
   - Open your browser and navigate to: `http://localhost:5000`
   - Enter a student ID from your test.csv file
   - Click "Predict Score" to see the predicted exam score

## Features

- Beautiful academic-themed UI design
- Real-time score prediction
- Animated score display
- Error handling for invalid IDs
- Automatic model training on first run (model is saved for subsequent runs)

## File Structure

- `app.py` - Flask application with model loading and prediction logic
- `templates/index.html` - Frontend HTML with academic theme
- `requirements.txt` - Python dependencies
- `model.xgb` - Saved XGBoost model (created after first run)
- `preprocessor.pkl` - Saved preprocessing pipeline (created after first run)
"# ComputerIntelleigence" 
