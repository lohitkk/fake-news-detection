document.getElementById("newsForm").addEventListener("submit", function(event) {
    event.preventDefault();

    const newsText = document.getElementById("newsInput").value;

    // Simulate predictions using all three models
    const predictions = predict(newsText);

    // Display the predictions
    let output = `
        <h2>Predictions:</h2>
        <p><strong>Random Forest Model:</strong> ${predictions.randomForest ? "Real" : "Fake"}</p>
        <p><strong>XGBoost Model:</strong> ${predictions.xgBoost ? "Real" : "Fake"}</p>
    `;
    document.getElementById("result").innerHTML = output;
});

// Simulated prediction function (replace this with actual backend calls if needed)
function predict(newsText) {
    // For demonstration, we're simulating predictions based on the presence of specific keywords
    const fakeKeywords = ["fake", "lies", "not true", "deception", "false"];
    
    // Simulate predictions based on the presence of keywords
    const isFake = fakeKeywords.some(keyword => newsText.toLowerCase().includes(keyword));

    return {
        randomForest: !isFake,  // Simulate Random Forest prediction
        xgBoost: !isFake        // Simulate XGBoost prediction
    };
}