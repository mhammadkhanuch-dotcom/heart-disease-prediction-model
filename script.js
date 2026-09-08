const form = document.getElementById("predictionForm");
const result = document.getElementById("result");

form.addEventListener("submit", async function(event) {

    event.preventDefault();

    // Collect form data
    const data = {
        age: parseInt(document.getElementById("age").value),
        sex: parseInt(document.getElementById("sex").value),
        cp: parseInt(document.getElementById("cp").value),
        trestbps: parseInt(document.getElementById("trestbps").value),
        chol: parseInt(document.getElementById("chol").value),
        fbs: parseInt(document.getElementById("fbs").value),
        restecg: parseInt(document.getElementById("restecg").value),
        thalach: parseInt(document.getElementById("thalach").value),
        exang: parseInt(document.getElementById("exang").value),
        oldpeak: parseFloat(document.getElementById("oldpeak").value),
        slope: parseInt(document.getElementById("slope").value),
        ca: parseInt(document.getElementById("ca").value),
        thal: parseInt(document.getElementById("thal").value),
    };

    // Send data to FastAPI backend
    try {
        const response = await fetch("/api/predict", {
            method: "POST",
            headers: {
                "Content-Type": "application/json",
            },
            body: JSON.stringify(data),
        });

        if (!response.ok) {
            const errorText = await response.text();
            throw new Error(`HTTP error! status: ${response.status}, message: ${errorText}`);
        }

        const resultData = await response.json();

        // Display the result
        result.innerHTML = `
            <h3>Prediction Result:</h3>
            <p><strong>Status:</strong> ${resultData.result}</p>
            <p><strong>Probability:</strong> ${resultData.probability}%</p>
        `;

    } catch (error) {
        console.error("Error:", error);
        result.innerHTML = `<p style="color: red;">Error making prediction: ${error.message}</p>`;
    }

});
