import os
import gc
import random
import inspect
import pickle
import numpy as np
import pandas as pd
from flask import Flask, render_template, request, jsonify
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
import xgboost as xgb

app = Flask(__name__)

# ---------------------------
# Reproducibility
# ---------------------------
def seed_everything(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except Exception:
        pass

seed_everything(42)

# ---------------------------
# Utilities
# ---------------------------
def stratify_bins(y, n_bins=10):
    y = pd.Series(y).astype(float).clip(lower=0)
    try:
        bins = pd.qcut(y, q=n_bins, labels=False, duplicates='drop')
    except Exception:
        bins = pd.cut(y, bins=n_bins, labels=False)
    return bins.fillna(0).astype(int)

def gpu_available():
    # Check for GPU on both Linux and Windows
    if os.path.exists("/proc/driver/nvidia/version"):
        return True
    try:
        import subprocess
        result = subprocess.run(['nvidia-smi'], capture_output=True, text=True)
        return result.returncode == 0
    except:
        return False

# ---------------------------
# Robust Feature Engineering
# ---------------------------
class RobustAcademicFE:
    def __init__(self, clip_quantiles=(0.01, 0.99)):
        self.clip_quantiles = clip_quantiles
        self._num_clip_bounds = {}
        self._drop_all_nan_cols = []
        self.sleep_map = {"very poor":1,"poor":2,"fair":3,"average":3,"okay":3,"good":4,"very good":5,"excellent":6}
        self.exam_diff_map = {"very easy":1,"easy":2,"medium":3,"hard":4,"very hard":5}
        self.facility_map  = {"very bad":1,"bad":2,"average":3,"good":4,"very good":5,"excellent":6}
        self._synonyms = {"ok":"okay","vg":"very good","v good":"very good","v bad":"very bad"}

    def _n(self, x):
        if pd.isna(x): return np.nan
        return str(x).strip().lower()

    def _map_ordinal(self, s, base_map, numeric_ok=True):
        def conv(v):
            vn = self._n(v)
            if vn is np.nan: return np.nan
            if vn in self._synonyms: vn = self._synonyms[vn]
            if numeric_ok:
                try: return float(vn)
                except Exception: pass
            return base_map.get(vn, np.nan)
        return s.map(conv)

    def _engineer(self, df):
        df = df.copy()
        if "sleep_quality" in df:
            df["sleep_quality_ord"] = self._map_ordinal(df["sleep_quality"], self.sleep_map)
        if "exam_difficulty" in df:
            df["exam_difficulty_ord"] = self._map_ordinal(df["exam_difficulty"], self.exam_diff_map)
        if "facility_rating" in df:
            df["facility_rating_ord"] = self._map_ordinal(df["facility_rating"], self.facility_map)
        if "study_hours" in df and "class_attendance" in df:
            df["study_x_attendance"] = df["study_hours"] * df["class_attendance"]
        if "sleep_hours" in df and "sleep_quality_ord" in df:
            df["sleep_effective"] = df["sleep_hours"] * df["sleep_quality_ord"]
        if "internet_access" in df:
            df["has_internet"] = df["internet_access"].map(
                lambda v: 0 if self._n(v) in ["no","none","null","nan",""] else 1
            )
        for col in ["study_hours","class_attendance","sleep_hours"]:
            if col in df:
                df[f"isna_{col}"] = df[col].isna().astype(int)
        return df

    def fit(self, X, y=None):
        Xb = self._engineer(X)
        num_cols = Xb.select_dtypes(include=["number"]).columns
        self._drop_all_nan_cols = [c for c in num_cols if Xb[c].notna().sum() == 0]
        ql, qh = self.clip_quantiles
        for col in ["study_hours","class_attendance","sleep_hours"]:
            if col in Xb:
                low, high = Xb[col].quantile([ql, qh])
                self._num_clip_bounds[col] = (float(low), float(high))
        return self

    def transform(self, X):
        Xb = self._engineer(X)
        Xb = Xb.drop(columns=self._drop_all_nan_cols, errors="ignore")
        for col, (low, high) in self._num_clip_bounds.items():
            if col in Xb:
                Xb[col] = Xb[col].clip(lower=low, upper=high)
        return Xb

    def fit_transform(self, X, y=None):
        return self.fit(X, y).transform(X)

# ---------------------------
# Model Loading/Training
# ---------------------------
def make_ohe():
    kwargs = {"handle_unknown":"ignore"}
    if "sparse_output" in inspect.signature(OneHotEncoder).parameters:
        kwargs["sparse_output"] = True
    else:
        kwargs["sparse"] = True
    return OneHotEncoder(**kwargs)

def train_and_save_model():
    """Train the model and save it along with preprocessor"""
    print("Training model...")
    
    train_path = "Dataset/train.csv"
    test_path = "test.csv"
    
    if not os.path.exists(train_path):
        train_path = "train.csv"
    if not os.path.exists(test_path):
        test_path = "Dataset/test.csv"
    
    if not os.path.exists(train_path):
        raise FileNotFoundError("train.csv not found")
    if not os.path.exists(test_path):
        raise FileNotFoundError("test.csv not found")
    
    df_train = pd.read_csv(train_path)
    df_test = pd.read_csv(test_path)
    
    TARGET = "exam_score"
    ID_COL = "id"
    EXCLUDE = {TARGET, ID_COL, "Transported"}
    
    df_train[TARGET] = pd.to_numeric(df_train[TARGET], errors="coerce")
    mask = df_train[TARGET].notna() & np.isfinite(df_train[TARGET].values) & (df_train[TARGET].abs() < 1e9)
    df_train = df_train.loc[mask].reset_index(drop=True)
    
    feature_cols = [c for c in df_train.columns if c not in EXCLUDE]
    X_full_raw = df_train[feature_cols].copy()
    y_full = df_train[TARGET].astype(float).values
    
    fe = RobustAcademicFE()
    X_full_fe = fe.fit_transform(X_full_raw)
    
    num_cols = X_full_fe.select_dtypes(include=["number"]).columns.tolist()
    cat_cols = X_full_fe.select_dtypes(include=["object","category","bool"]).columns.tolist()
    
    pre = ColumnTransformer(
        transformers=[
            ("num", SimpleImputer(strategy="median"), num_cols),
            ("cat", Pipeline([
                ("imp", SimpleImputer(strategy="most_frequent")),
                ("ohe", make_ohe())
            ]), cat_cols),
        ],
        remainder="drop",
        sparse_threshold=1.0
    )
    
    bins = stratify_bins(y_full, n_bins=10)
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.1, random_state=42)
    tr_idx, va_idx = next(sss.split(X_full_fe, bins))
    
    X_tr, X_va = X_full_fe.iloc[tr_idx], X_full_fe.iloc[va_idx]
    y_tr, y_va = y_full[tr_idx], y_full[va_idx]
    
    X_tr_enc = pre.fit_transform(X_tr)
    X_va_enc = pre.transform(X_va)
    
    use_gpu = gpu_available()
    params = {
        "objective": "reg:squarederror",
        "eval_metric": "mae",
        "tree_method": "gpu_hist" if use_gpu else "hist",
        "predictor": "gpu_predictor" if use_gpu else "auto",
        "learning_rate": 0.03,
        "max_depth": 6,
        "min_child_weight": 2,
        "subsample": 0.85,
        "colsample_bytree": 0.75,
        "lambda": 1.0,
        "random_state": 42,
        "verbosity": 0
    }
    
    dtr = xgb.DMatrix(X_tr_enc, label=y_tr)
    dva = xgb.DMatrix(X_va_enc, label=y_va)
    
    evals = [(dtr, "train"), (dva, "valid")]
    bst = xgb.train(
        params=params,
        dtrain=dtr,
        num_boost_round=3500,
        evals=evals,
        early_stopping_rounds=200,
        verbose_eval=False
    )
    
    # Save model and preprocessor
    bst.save_model("model.xgb")
    with open("preprocessor.pkl", "wb") as f:
        pickle.dump({"fe": fe, "pre": pre, "feature_cols": feature_cols, "common_cols": sorted(X_full_fe.columns)}, f)
    
    print("Model saved successfully!")
    return bst, fe, pre, feature_cols, sorted(X_full_fe.columns)

def load_model():
    """Load model and preprocessor if they exist, otherwise train"""
    if os.path.exists("model.xgb") and os.path.exists("preprocessor.pkl"):
        print("Loading saved model...")
        bst = xgb.Booster()
        bst.load_model("model.xgb")
        with open("preprocessor.pkl", "rb") as f:
            data = pickle.load(f)
        return bst, data["fe"], data["pre"], data["feature_cols"], data["common_cols"]
    else:
        return train_and_save_model()

# Load model on startup
print("Initializing model...")
model, fe, preprocessor, feature_cols, common_cols = load_model()
print("Model ready!")

# ---------------------------
# Flask Routes
# ---------------------------
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        student_id = data.get('id')
        
        if student_id is None:
            return jsonify({'error': 'ID is required'}), 400
        
        # Load test data
        test_path = "test.csv"
        if not os.path.exists(test_path):
            test_path = "Dataset/test.csv"
        
        if not os.path.exists(test_path):
            return jsonify({'error': 'test.csv not found'}), 404
        
        df_test = pd.read_csv(test_path)
        
        # Find student by ID
        student_row = df_test[df_test['id'] == int(student_id)]
        
        if student_row.empty:
            return jsonify({'error': f'Student ID {student_id} not found'}), 404
        
        # Extract features
        X_test_raw = student_row[feature_cols].copy()
        
        # Feature engineering
        X_test_fe = fe.transform(X_test_raw)
        
        # Ensure same columns as training
        X_test_fe = X_test_fe[[c for c in common_cols if c in X_test_fe.columns]].copy()
        
        # Preprocess
        X_test_enc = preprocessor.transform(X_test_fe)
        
        # Predict
        dtest = xgb.DMatrix(X_test_enc)
        try:
            pred = model.predict(dtest, iteration_range=(0, model.best_iteration + 1))[0]
        except Exception:
            pred = model.predict(dtest, ntree_limit=model.best_ntree_limit)[0]
        
        # Clip prediction
        pred = max(0.0, float(pred))
        
        return jsonify({
            'success': True,
            'id': int(student_id),
            'exam_score': round(pred, 2)
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
