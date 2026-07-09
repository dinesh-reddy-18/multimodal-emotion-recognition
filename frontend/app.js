document.addEventListener('DOMContentLoaded', () => {
    // DOM Elements
    const tabBtns = document.querySelectorAll('.tab-btn');
    const tabContents = document.querySelectorAll('.tab-content');
    const videoUpload = document.getElementById('video-upload-area');
    const videoInput = document.getElementById('video-file-input');
    const videoName = document.getElementById('video-name');
    const audioUpload = document.getElementById('audio-upload-area');
    const audioInput = document.getElementById('audio-file-input');
    const audioName = document.getElementById('audio-name');
    const textInput = document.getElementById('text-input-field');
    const btnPredict = document.getElementById('btn-predict');
    const loader = document.getElementById('loader');
    
    const apiStatusDot = document.getElementById('api-status-dot');
    const apiStatusText = document.getElementById('api-status-text');
    
    const resultsIdle = document.getElementById('results-idle-state');
    const resultsActive = document.getElementById('results-active-state');
    
    const fusedEmotionLabel = document.getElementById('fused-emotion-label');
    const fusedConfidence = document.getElementById('fused-confidence');
    
    const cardAudio = document.getElementById('card-audio-summary');
    const audioLabel = document.getElementById('audio-prediction-label');
    const audioConf = document.getElementById('audio-prediction-conf');
    
    const cardFace = document.getElementById('card-face-summary');
    const faceLabel = document.getElementById('face-prediction-label');
    const faceConf = document.getElementById('face-prediction-conf');
    
    const cardText = document.getElementById('card-text-summary');
    const textLabel = document.getElementById('text-prediction-label');
    const textConf = document.getElementById('text-prediction-conf');
    
    let chartInstance = null;
    let activeTab = 'video';
    
    // Check API server connection on startup
    checkApiStatus();

    // Tab Switching Navigation
    tabBtns.forEach(btn => {
        btn.addEventListener('click', () => {
            tabBtns.forEach(b => b.classList.remove('active'));
            tabContents.forEach(c => c.style.display = 'none');
            
            btn.classList.add('active');
            activeTab = btn.getAttribute('data-tab');
            document.getElementById(`tab-${activeTab}`).style.display = 'block';
        });
    });

    // Upload Click Triggers
    videoUpload.addEventListener('click', () => videoInput.click());
    audioUpload.addEventListener('click', () => audioInput.click());

    // File Selection Handlers
    videoInput.addEventListener('change', (e) => {
        if (e.target.files.length > 0) {
            videoName.textContent = e.target.files[0].name;
            videoUpload.querySelector('.upload-instructions').textContent = "File selected successfully.";
        }
    });

    audioInput.addEventListener('change', (e) => {
        if (e.target.files.length > 0) {
            audioName.textContent = e.target.files[0].name;
            audioUpload.querySelector('.upload-instructions').textContent = "File selected successfully.";
        }
    });

    // Main API Call prediction trigger
    btnPredict.addEventListener('click', async () => {
        const formData = new FormData();
        let hasInput = false;

        if (activeTab === 'video') {
            if (videoInput.files.length > 0) {
                formData.append('video', videoInput.files[0]);
                hasInput = true;
            } else {
                alert("Please select a video file first.");
                return;
            }
        } else if (activeTab === 'audio') {
            if (audioInput.files.length > 0) {
                formData.append('audio', audioInput.files[0]);
                hasInput = true;
            } else {
                alert("Please select an audio file first.");
                return;
            }
        } else if (activeTab === 'text') {
            const val = textInput.value.trim();
            if (val) {
                formData.append('text', val);
                hasInput = true;
            } else {
                alert("Please type a statement first.");
                return;
            }
        }

        if (!hasInput) return;

        // Toggle Loading State
        loader.style.display = 'flex';
        resultsIdle.style.display = 'flex';
        resultsActive.style.display = 'none';

        try {
            const response = await fetch('/api/predict', {
                method: 'POST',
                body: formData
            });

            if (!response.ok) {
                const parseErr = await response.json();
                throw new Error(parseErr.detail || "Server error occurred");
            }

            const data = await response.json();
            displayPredictions(data);
        } catch (error) {
            alert(`Analysis failed: ${error.message}`);
        } finally {
            loader.style.display = 'none';
        }
    });

    // Parse status checker
    async function checkApiStatus() {
        try {
            const res = await fetch('/api/models/status');
            if (res.ok) {
                const info = await res.ok ? await res.json() : {};
                apiStatusDot.className = 'indicator-dot green';
                apiStatusText.textContent = 'Models Online';
            } else {
                apiStatusDot.className = 'indicator-dot red';
                apiStatusText.textContent = 'Server Error';
            }
        } catch (e) {
            apiStatusDot.className = 'indicator-dot red';
            apiStatusText.textContent = 'Server Offline';
        }
    }

    // Display predictions function
    function displayPredictions(data) {
        // Toggle Panel displays
        resultsIdle.style.display = 'none';
        resultsActive.style.display = 'block';

        // 1. Fused banner
        const fused = data.final_prediction;
        fusedEmotionLabel.textContent = capitalizeFirstLetter(fused.label);
        fusedConfidence.textContent = `${(fused.confidence * 100).toFixed(1)}%`;

        // 2. Modality Cards logic
        const mods = data.modalities;
        
        // Audio
        if (mods.audio) {
            cardAudio.classList.remove('inactive');
            audioLabel.textContent = capitalizeFirstLetter(mods.audio.label);
            audioConf.textContent = `Confidence: ${(mods.audio.confidence * 100).toFixed(1)}%`;
        } else {
            cardAudio.classList.add('inactive');
            audioLabel.textContent = "Unavailable";
            audioConf.textContent = "Confidence: N/A";
        }

        // Face / Video
        if (mods.face_video) {
            cardFace.classList.remove('inactive');
            faceLabel.textContent = capitalizeFirstLetter(mods.face_video.label);
            faceConf.textContent = `Confidence: ${(mods.face_video.confidence * 100).toFixed(1)}%`;
        } else {
            cardFace.classList.add('inactive');
            faceLabel.textContent = "Unavailable";
            faceConf.textContent = "Confidence: N/A";
        }

        // Text
        if (mods.text) {
            cardText.classList.remove('inactive');
            textLabel.textContent = capitalizeFirstLetter(mods.text.label);
            textConf.textContent = `Confidence: ${(mods.text.confidence * 100).toFixed(1)}%`;
        } else {
            cardText.classList.add('inactive');
            textLabel.textContent = "Unavailable";
            textConf.textContent = "Confidence: N/A";
        }

        // 3. Render Chart
        renderChart(data);
    }

    function renderChart(data) {
        const labels = ['Angry', 'Disgust', 'Fearful', 'Happy', 'Neutral', 'Sad', 'Surprised'];
        
        // Collect active chart datasets
        const datasets = [];
        const mods = data.modalities;
        
        if (mods.audio) {
            datasets.push({
                label: 'Audio Modality',
                data: mods.audio.probs,
                backgroundColor: 'rgba(99, 102, 241, 0.45)',
                borderColor: 'rgba(99, 102, 241, 1)',
                borderWidth: 1.5
            });
        }
        
        if (mods.face_video) {
            datasets.push({
                label: 'Facial Expression',
                data: mods.face_video.probs,
                backgroundColor: 'rgba(236, 72, 153, 0.45)',
                borderColor: 'rgba(236, 72, 153, 1)',
                borderWidth: 1.5
            });
        }
        
        if (mods.text) {
            datasets.push({
                label: 'Text Context',
                data: mods.text.probs,
                backgroundColor: 'rgba(234, 179, 8, 0.45)',
                borderColor: 'rgba(234, 179, 8, 1)',
                borderWidth: 1.5
            });
        }
        
        // Add Fused dataset prediction if present
        const fuse_active = data.fusion.late_fusion || data.fusion.early_fusion;
        if (fuse_active) {
            datasets.push({
                label: 'Fused Prediction',
                data: fuse_active.probs,
                backgroundColor: 'rgba(139, 92, 246, 0.8)',
                borderColor: 'rgba(139, 92, 246, 1)',
                borderWidth: 2,
                borderRadius: 4
            });
        }

        const ctx = document.getElementById('probsChart').getContext('2d');
        
        // Destroy old Chart instance to prevent rendering conflicts
        if (chartInstance) {
            chartInstance.destroy();
        }

        chartInstance = new Chart(ctx, {
            type: 'bar',
            data: { labels, datasets },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                scales: {
                    x: {
                        grid: { color: 'rgba(255, 255, 255, 0.05)' },
                        ticks: { color: '#9ca3af', font: { family: 'Inter', size: 11 } }
                    },
                    y: {
                        min: 0,
                        max: 1.0,
                        grid: { color: 'rgba(255, 255, 255, 0.05)' },
                        ticks: { color: '#9ca3af', font: { family: 'Inter', size: 11 } }
                    }
                },
                plugins: {
                    legend: {
                        labels: { color: '#f3f4f6', font: { family: 'Inter', weight: 'medium' } }
                    },
                    tooltip: {
                        backgroundColor: '#141623',
                        titleFont: { family: 'Outfit', weight: 'bold' },
                        bodyFont: { family: 'Inter' },
                        padding: 10,
                        borderWidth: 1,
                        borderColor: 'rgba(255, 255, 255, 0.08)'
                    }
                }
            }
        });
    }

    function capitalizeFirstLetter(string) {
        if (!string) return '';
        if (string.toLowerCase() === 'fearful') return 'Fearful';
        if (string.toLowerCase() === 'surprised') return 'Surprised';
        return string.charAt(0).toUpperCase() + string.slice(1);
    }
});
