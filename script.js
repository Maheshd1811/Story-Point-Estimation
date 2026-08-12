document.addEventListener('DOMContentLoaded', () => {
    const form = document.getElementById('estimation-form');
    const input = document.getElementById('story-input');
    const submitBtn = document.getElementById('submit-btn');
    const loadingState = document.getElementById('loading-state');
    const resultState = document.getElementById('result-state');
    const errorState = document.getElementById('error-state');
    const errorMessage = document.getElementById('error-message');
    const resultPoints = document.getElementById('result-points');
    const resultConfidence = document.getElementById('result-confidence');

    form.addEventListener('submit', async (e) => {
        e.preventDefault();
        const text = input.value.trim();
        
        if (!text) return;

        // Hide old results/errors
        resultState.classList.add('hidden');
        errorState.classList.add('hidden');
        
        // Show loading state
        loadingState.classList.remove('hidden');
        submitBtn.disabled = true;

        try {
            // Fetch JSON from the backend
            const response = await fetch('http://localhost:8000/predict', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({ text: text })
            });

            if (!response.ok) {
                const errData = await response.json().catch(() => ({}));
                throw new Error(errData.detail || `Server Error: ${response.status}`);
            }

            const data = await response.json();
            
            // Populate results
            resultPoints.textContent = data.predicted_story_points || 'N/A';
            
            // Capitalize confidence
            let conf = data.confidence || 'High';
            resultConfidence.textContent = conf.charAt(0).toUpperCase() + conf.slice(1);
            
            // Show result
            resultState.classList.remove('hidden');
            
            // Trigger purely CSS balloons effect
            createBalloons();

        } catch (error) {
            errorMessage.textContent = `❌ ${error.message}`;
            errorState.classList.remove('hidden');
        } finally {
            // Hide loading state
            loadingState.classList.add('hidden');
            submitBtn.disabled = false;
        }
    });

    // Helper for CSS balloon animation
    function createBalloons() {
        const container = document.createElement('div');
        container.className = 'balloon-container';
        document.body.appendChild(container);
        
        const emojis = ['🎈', '🎊', '🎉', '🚀', '✨'];
        
        for (let i = 0; i < 15; i++) {
            const balloon = document.createElement('div');
            balloon.className = 'balloon';
            balloon.textContent = emojis[Math.floor(Math.random() * emojis.length)];
            balloon.style.left = `${Math.random() * 100}vw`;
            balloon.style.animationDelay = `${Math.random() * 0.5}s`;
            balloon.style.fontSize = `${1.5 + Math.random() * 1.5}rem`;
            container.appendChild(balloon);
        }
        
        // Clean up DOM after animation finishes (4s)
        setTimeout(() => {
            if (document.body.contains(container)) {
                document.body.removeChild(container);
            }
        }, 5000);
    }
});
