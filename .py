from pathlib import Path 
import json
import joblib 
import pandas as pd     
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier 
from sklearn.impute import SimpleImputer
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score   
from sklearn.model_selection import train_test_split    
from sklearn.pipeline import Pipeline 
from sklearn.preprocessing import OneHotEncoder, StandardScaler 

HERE = Path(__file__).resolve().parent

DATA_PATH = HERE /'ml_data'/'telecom_churn.csv'
MODEL_PATH = HERE / 'churn_model.joblib'
METRICS_PATH = HERE / 'metrics.json'

FEATURES = ['usage_minutes' , 'complaints', 'contract_months', 'monthly_fee', 'contract_type', 'region']

TARGET = 'churn'

NUMERIC = ['usage_minutes' , 'complaints', 'contract_months', 'monthly_fee']
CATEGORICAL = ['contract_type', 'region']


def build_pipeline() -> Pipeline:

    numeric_pipe = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    category_pipe = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    preprocessor = ColumnTransformer([
        ('numeric', numeric_pipe, NUMERIC),
        ('category', category_pipe, CATEGORICAL)
    ])

    model = RandomForestClassifier(
        n_estimators=300,
        class_weight='balanced', 
        random_state=42,
        min_samples_leaf=3 
    )

    return Pipeline([('preprocess', preprocessor), ('model', model)])


def train() -> dict:
    """재현 가능한 분할로 모델을 학습하고, 실무 설명용 지표를 반환한다."""

    data = pd.read_csv(DATA_PATH)

    X = data[FEATURES]
    y = data[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    Pipeline = build_pipeline()
    Pipeline.fit(X_train, y_train)

    probabilities = Pipeline.predict_proba(X_test)[:,1]

    predictions = (probabilities >= 0.5).astype(int) # int형변환

    report = classification_report(y_test, predictions, output_dict=True, zero_division=0)

    # 보고용으로 핵심 자료만 정리
    metrics = {
        'test_rows' : len(X_test),
        'roc_auc' : round(float(roc_auc_score(y_test, probabilities)), 4),
        'recall_churn' : round(float(report['1']['recall']), 4),
        'precision_churn': round(float(report['1']['precision']),4),
        'confusion_matrix': confusion_matrix(y_test, predictions).tolist()
    }

    joblib.dump(Pipeline, MODEL_PATH)

    METRICS_PATH.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding='utf-8-sig')

    return metrics

if __name__ == '__main__':
    print(json.dumps(train(), ensure_ascii=False,indent=2))