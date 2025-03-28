document.addEventListener('DOMContentLoaded', () => {
    const carForm = document.getElementById('carForm');
    const resultContainer = document.getElementById('result');
    const predictedPrice = document.getElementById('predictedPrice');
    const resetButton = document.getElementById('resetButton');

    carForm.addEventListener('submit', async (e) => {
        e.preventDefault();

        // Get form values
        const formData = {
            mileage: parseFloat(document.getElementById('mileage').value),
            age: parseInt(document.getElementById('age').value),
            engineSize: parseFloat(document.getElementById('engineSize').value),
            horsepower: parseInt(document.getElementById('horsepower').value),
            fuelEfficiency: parseFloat(document.getElementById('fuelEfficiency').value)
        };

        try {
            // Show loading state
            const predictButton = carForm.querySelector('.predict-button');
            predictButton.disabled = true;
            predictButton.textContent = 'Predicting...';

            // Make API call
            const response = await fetch('http://localhost:8000/predict', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    features: [[
                        formData.mileage,
                        formData.age,
                        formData.engineSize,
                        formData.horsepower,
                        formData.fuelEfficiency
                    ]]
                })
            });

            if (!response.ok) {
                throw new Error('Prediction failed');
            }

            const data = await response.json();
            const price = data.predictions[0];

            // Update UI with result
            predictedPrice.textContent = `$${price.toLocaleString(undefined, {
                minimumFractionDigits: 2,
                maximumFractionDigits: 2
            })}`;
            resultContainer.classList.remove('hidden');

            // Scroll to result
            resultContainer.scrollIntoView({ behavior: 'smooth' });

        } catch (error) {
            alert('Error making prediction. Please try again.');
            console.error('Prediction error:', error);
        } finally {
            // Reset button state
            const predictButton = carForm.querySelector('.predict-button');
            predictButton.disabled = false;
            predictButton.textContent = 'Predict Price';
        }
    });

    resetButton.addEventListener('click', () => {
        // Reset form
        carForm.reset();
        
        // Hide result
        resultContainer.classList.add('hidden');
        
        // Scroll to form
        carForm.scrollIntoView({ behavior: 'smooth' });
    });
}); 