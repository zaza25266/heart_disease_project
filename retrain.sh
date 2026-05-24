#!/bin/bash

# exit immediately if command fails
set -e
# Tell Python to use the current directory as the root for imports
export PYTHONPATH=$(pwd)

echo "WARNING: you are about to retrain model ..."

read -p "Are you sure to retrain model (y/n): " confirm

if [[ "$confirm" != "y" && "$confirm" != "Y" ]]; then
    echo "Retraining is canceled"
    exit 0
fi

if [[ "$OSTYPE" == "msys" || "$OSTYPE" == "cygwin" || "$OSTYPE" == "win32" ]]; then
    echo "window OS detected (Git Bash)"
    PYTHON_CMD="python"
    VENV_ACTIVATE="source venv/Scripts/activate"
else
    echo "mac/linux OS detected"
    PYTHON_CMD="python3"
    VENV_ACTIVATE="source venv/bin/activate"
fi

echo "retraining model ..."

echo "[1/4] checking virtual environment ..."

if [ ! -d "venv" ] || [ ! -f "venv/.installed" ]; then
    echo "virtual environment not found! creating new ...!"
    $PYTHON_CMD -m venv venv
    $VENV_ACTIVATE

    echo "installing dependencies from requirements.txt..."
    if [ ! -f "local_requirements.txt" ]; then
        echo "local_requirements.txt not found ..."
        exit 1
    fi 

    pip install --upgrade pip
    pip install -r local_requirements.txt 

    touch venv/.installed # create hidden file to show if dependencies already created
else
    $VENV_ACTIVATE
fi
   

echo "[2/4] running data_setup.py ..."
$PYTHON_CMD  setup/data_setup.py

echo "[3/4] running train.py ..."
echo "Booting MLflow Tracking Server in a new window..."
osascript -e 'tell app "Terminal" to do script "cd '$PWD' && source venv/bin/activate && mlflow ui --port 5001"'
$PYTHON_CMD  training/train.py

echo "[4/4] running py_tests ..."
pytest py_test/

echo "new model is trained, tested, and ready..."


