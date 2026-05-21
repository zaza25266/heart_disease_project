document.getElementById('prediction-form').addEventListener('submit', async function(e) {
    e.preventDefault(); // Stop the page from reloading

    // 1. Gather all the data from the form
    const payload = {
        age: parseFloat(document.getElementById('age').value),
        trestbps: parseFloat(document.getElementById('trestbps').value),
        chol: parseFloat(document.getElementById('chol').value),
        thalach: parseFloat(document.getElementById('thalach').value),
        oldpeak: parseFloat(document.getElementById('oldpeak').value),
        cp: document.getElementById('cp').value,
        restecg: document.getElementById('restecg').value,
        slope: document.getElementById('slope').value,
        thal: document.getElementById('thal').value
    };

    // 2. Setup the Result Box UI
    const resultBox = document.getElementById('result-box');
    const resultMessage = document.getElementById('result-message');
    const systemMessage = document.getElementById('system-message');
    
    // Show the box and trigger the custom CSS fade-in animation
    resultBox.classList.remove('hidden');
    resultBox.classList.add('fade-in');
    
    resultMessage.className = "p-4 rounded-md font-bold text-lg mb-2 bg-gray-200 text-gray-700";
    resultMessage.innerText = "Transmitting to ML Backend...";
    systemMessage.innerText = "";

    try {
        // 3. Send the data to your FastAPI server
        const response = await fetch('http://localhost:8000/predict', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(payload)
        });

        if (response.ok) {
            const data = await response.json();
            const confidence = (data.risk_probability * 100).toFixed(1);

            // 4. Update the UI based on the prediction
            if (data.prediction_class === 1) {
                resultMessage.className = "p-4 rounded-md font-bold text-lg mb-2 bg-red-100 text-red-800 border border-red-300";
                resultMessage.innerText = `${data.clinical_status} (Confidence: ${confidence}%)`;
            } else {
                resultMessage.className = "p-4 rounded-md font-bold text-lg mb-2 bg-green-100 text-green-800 border border-green-300";
                resultMessage.innerText = `${data.clinical_status} (Confidence: ${confidence}%)`;
            }
            systemMessage.innerText = `System Message: ${data.message}`;
        } else {
            const errorData = await response.json();
            resultMessage.className = "p-4 rounded-md font-bold text-lg mb-2 bg-yellow-100 text-yellow-800 border border-yellow-300";
            resultMessage.innerText = `API Error: ${errorData.detail || response.statusText}`;
        }
    } catch (error) {
        resultMessage.className = "p-4 rounded-md font-bold text-lg mb-2 bg-red-100 text-red-800 border border-red-300";
        resultMessage.innerText = "Failed to connect to the backend API. Is FastAPI running?";
    }
});