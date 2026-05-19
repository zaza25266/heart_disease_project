
import pytest 
import numpy as np 
from preprosess_pipeline.pre_processing import load_and_process_data, build_preprocessing_pipeline
from sklearn.ensemble import RandomForestClassifier

@pytest.fixture
def data_split():
    return load_and_process_data()

@pytest.fixture
def pipelines():
    return build_preprocessing_pipeline()


def test_data_splits_are_correct(data_split):
    """
    Test 1: ensures the database extraction and train/test splits return
    the correct data shapes and that no target data is missing  
    """
    X_train, X_test, y_train, y_test = data_split
    
    assert len(X_train) > len(X_test), "Training set should be larger than test set"
    assert X_train.shape[1] == 14, "There should be 14 features in the training data"
    assert y_train.isnull().sum() == 0, "There should be no missing values in the training target data"
    assert y_test.isnull().sum() == 0, "There should be no missing"
    
    unique_targets = y_train.unique()
    assert set(unique_targets) == {0, 1, 2, 3, 4}, "Target variable should have values from 0 to 4"
    
def test_preprocessing_pipeline(data_split, pipelines):
    """
    Test 2: Ensures the Scikit-Learn pipeline successfully handles
    imputation, scaling, encoding, and feature selection without crashing.
    """
    X_train, X_test, y_train, y_test = data_split
    pipeline = pipelines
    
    X_train_processed = pipeline.fit_transform(X_train, y_train)
    assert X_train_processed.shape[0] == X_train.shape[0], "Number of samples should remain the same after preprocessing"
    assert not np.isnan(X_train_processed).any(), "There should be no NaN values in the processed training data"
    
    assert X_train_processed.shape[1] <= 10, "Feature selection should reduce the number of features to 10 or fewer"
    

def test_no_data_leakage(data_split):
    """
    Test 3: Data Leakage Test.
    Ensures that absolutely zero rows from the test set accidentally bled into the train set.
    """
    X_train, X_test, _, _ = data_split
    common_indices = X_train.index.intersection(X_test.index)
    assert len(common_indices) == 0, "There should be no common indices between train and test sets"
    
    
def test_model_training_smoke_test(data_split, pipelines):
    """
    Test 4: Model Smoke Test.
    Proves that the fully processed data can successfully train a real ML model 
    and generate predictions without throwing shape or datatype crashes.
    """
    X_train, X_test, y_train, y_test = data_split
    pipeline = pipelines
    
    # Process both train and test data
    X_train_processed = pipeline.fit_transform(X_train, y_train)
    X_test_processed = pipeline.transform(X_test)
    
    # dummy model to test the processed data
    model = RandomForestClassifier(n_estimators=5, random_state=42)
    
    try:
        model.fit(X_train_processed, y_train)
        predictions = model.predict(X_test_processed)
    except Exception as e:
        pytest.fail(f"Model training or prediction failed with error: {e}")
        
    assert len(predictions) == len(X_test), "Number of predictions should match number of test samples"
    

def test_pipeline_determinism(data_splits, pipelines):
    """
    Test 5: Pipeline determinism
    ensures that running the pipeline twice on the same data yields 
    the EXACT same results
    """
    X_train, X_test, y_train, y_test = data_splits
    
    pipeline1 = pipelines
    pipeline2 = pipelines
    
    output1 = pipeline1.fit_transform(X_train, y_train)
    output2 = pipeline2.fit_transform(X_train, y_train)
    
    # np.testing will instantly crash the test if even a single decimal is different
    np.testing.assert_array_almost_equal(
        output1, output2, 
        err_msg="CRITICAL: Pipeline is not deterministic! Random states might be missing."
    )
    

def test_stratification_balance(data_splits):
    """
    Test 6: class stratification check
    ensures that the ratio of sick vs healthy patients is mathematically preserved 
    across both the training and testing sets
    """
    _, _, y_train, y_test = data_splits
    
    # Calculate the percentage of positive cases (1s) in both sets
    train_sick_ratio = y_train.mean()
    test_sick_ratio = y_test.mean()
    
    # The ratios should be nearly identical (within a 5% margin of error)
    difference = np.abs(train_sick_ratio - test_sick_ratio)
    assert difference < 0.05, f"Stratification failed! Train ratio: {train_sick_ratio:.2f}, Test ratio: {test_sick_ratio:.2f}"
    
    
    