document.getElementById("churnForm").addEventListener("submit", async function(event) {
    event.preventDefault();

    let data = {
        credit_score: document.getElementById("credit_score").value,
        age: document.getElementById("age").value,
        balance: document.getElementById("balance").value,
        num_products: document.getElementById("num_products").value
    };

    let response = await fetch("/predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(data)
    });

    let result = await response.json();
    let resultDiv = document.getElementById("result");

    if (result.error) {
        resultDiv.innerHTML = `<p style="color: red;">⚠️ Error: ${result.error}</p>`;
    } else {
        resultDiv.innerHTML = `<p>${result.prediction}</p>`;
    }
});
