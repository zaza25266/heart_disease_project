
import joblib
import pandas as pd
from preprosess_pipeline.pre_processing import load_and_process_data

def test_saved_model():
    print('loading raw data')
    _, X_test, _, y_test = load_and_process_data()
    
    # grab 5 patients 
    sample_patients = X_test.head(5)
    actual_outcomes = y_test.head(5).values
    print('loading model')
    try:
        model = joblib.load("best_Tuned_Logistic_Regression.pkl")
    except FileNotFoundError:
        print("model not found")
        return
    
    print("making prediction")
    predictions = model.predict(sample_patients)
    probabilities = model.predict_proba(sample_patients)[:, 1]

    print('result of 5 test patients:')
    
    for i in range(5):
        prob_pct = probabilities[i] * 100
        pred_label = "Sick (1)" if predictions[i] == 1 else "Healthy (0)"
        true_label = "Sick (1)" if actual_outcomes[i] == 1 else "Healthy (0)"
        
        # Check if the model got it right
        status = "CORRECT" if predictions[i] == actual_outcomes[i] else "WRONG"
        
        print(f"Patient {i+1}:")
        print(f"  -> Model Predicts: {pred_label} (Confidence: {prob_pct:.1f}%)")
        print(f"  -> Actual Truth:   {true_label}")
        print(f"  -> Result:         {status}\n")
        
        
if __name__ == "__main__":
    test_saved_model()