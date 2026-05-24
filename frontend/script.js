// guarantees the HTML is fully loaded before JS attaches
document.addEventListener('DOMContentLoaded', () => {

    const form = document.getElementById('prediction-form');
    const loading = document.getElementById('loading-message');
    const resultBox = document.getElementById('result-box');
    const resultMessage = document.getElementById('result-message');
    const systemMessage = document.getElementById('system-message');
    const exportBtn = document.getElementById('export-btn');

    form.addEventListener('submit', async function (e) {
        e.preventDefault();

        // UI reset
        loading.classList.remove('hidden');
        resultBox.classList.add('hidden');

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

        try {
            const response = await fetch('https://heart-disease-api-1mrb.onrender.com/predict', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(payload)
            });

            resultBox.classList.remove('hidden');

            if (!response.ok) {
                const err = await response.json();
                resultMessage.className = "p-4 rounded-md font-bold text-lg mb-2 bg-yellow-100 text-yellow-800 border border-yellow-300";
                resultMessage.innerText = `API Error: ${err.detail || response.statusText}`;
                return;
            }

            const data = await response.json();

            const probability = (data.risk_probability * 100).toFixed(1);
            const isHighRisk = data.prediction_class === 1;

            resultMessage.className = isHighRisk
                ? "p-4 rounded-md font-bold text-lg mb-2 bg-red-100 text-red-800 border border-red-300"
                : "p-4 rounded-md font-bold text-lg mb-2 bg-green-100 text-green-800 border border-green-300";

            resultMessage.innerText =
`${data.clinical_status}
Disease Probability: ${probability}%`;

            const date = new Date().toLocaleString();
            systemMessage.innerText = `Report Generated: ${date}\nSystem: ${data.message}`;

        } catch (error) {
            resultMessage.className = "p-4 rounded-md font-bold text-lg mb-2 bg-red-100 text-red-800 border border-red-300";
            resultMessage.innerText = "Failed to connect to the backend API.";
        } finally {
            loading.classList.add('hidden');
        }
    });

    // PDF export
    exportBtn.addEventListener('click', () => {
        window.print();
    });

});