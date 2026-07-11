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
    
    // Loader Elements
    const loader = document.getElementById('loader');
    const stepUpload = document.getElementById('step-upload');
    const stepDetect = document.getElementById('step-detect');
    const stepExtract = document.getElementById('step-extract');
    const stepFuse = document.getElementById('step-fuse');
    
    const apiStatusDot = document.getElementById('api-status-dot');
    const apiStatusText = document.getElementById('api-status-text');
    
    const resultsIdle = document.getElementById('results-idle-state');
    const resultsActive = document.getElementById('results-active-state');
    
    const fusedEmotionLabel = document.getElementById('fused-emotion-label');
    const fusedConfidence = document.getElementById('fused-confidence');
    const fusedProgressBar = document.getElementById('fused-progress-bar');
    
    // Modality Cards & Bars
    const cardAudio = document.getElementById('card-audio-summary');
    const audioLabel = document.getElementById('audio-prediction-label');
    const audioConf = document.getElementById('audio-prediction-conf');
    const barAudio = document.getElementById('bar-audio');
    
    const cardFace = document.getElementById('card-face-summary');
    const faceLabel = document.getElementById('face-prediction-label');
    const faceConf = document.getElementById('face-prediction-conf');
    const barFace = document.getElementById('bar-face');
    
    const cardText = document.getElementById('card-text-summary');
    const textLabel = document.getElementById('text-prediction-label');
    const textConf = document.getElementById('text-prediction-conf');
    const barText = document.getElementById('bar-text');
    
    let chartInstance = null;
    let activeTab = 'video';
    let recordedVideoFile = null;
    let recordedAudioFile = null;
    
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
            } else if (recordedVideoFile) {
                formData.append('video', recordedVideoFile);
                hasInput = true;
            } else {
                alert("Please select a video file or record using live webcam first.");
                return;
            }
        } else if (activeTab === 'audio') {
            if (audioInput.files.length > 0) {
                formData.append('audio', audioInput.files[0]);
                hasInput = true;
            } else if (recordedAudioFile) {
                formData.append('audio', recordedAudioFile);
                hasInput = true;
            } else {
                alert("Please select an audio file or record using microphone first.");
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

        // Reset Loader steps
        loader.style.display = 'flex';
        resultsIdle.style.display = 'flex';
        resultsActive.style.display = 'none';
        
        stepUpload.className = "step-item active";
        stepDetect.className = "step-item";
        stepExtract.className = "step-item";
        stepFuse.className = "step-item";
        
        // Progressive loading steps simulation
        const step1Timer = setTimeout(() => {
            stepUpload.className = "step-item done";
            stepDetect.className = "step-item active";
        }, 600);
        
        const step2Timer = setTimeout(() => {
            stepDetect.className = "step-item done";
            stepExtract.className = "step-item active";
        }, 1200);

        const step3Timer = setTimeout(() => {
            stepExtract.className = "step-item done";
            stepFuse.className = "step-item active";
        }, 2200);

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
            
            // Mark all steps as complete
            clearTimeout(step1Timer);
            clearTimeout(step2Timer);
            clearTimeout(step3Timer);
            
            stepUpload.className = "step-item done";
            stepDetect.className = "step-item done";
            stepExtract.className = "step-item done";
            stepFuse.className = "step-item done";
            
            setTimeout(() => {
                displayPredictions(data);
                loader.style.display = 'none';
            }, 300);
        } catch (error) {
            clearTimeout(step1Timer);
            clearTimeout(step2Timer);
            clearTimeout(step3Timer);
            loader.style.display = 'none';
            alert(`Analysis failed: ${error.message}`);
        }
    });

    // Parse status checker
    async function checkApiStatus() {
        try {
            const res = await fetch('/api/models/status');
            if (res.ok) {
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
        const fusedPct = (fused.confidence * 100).toFixed(1);
        fusedConfidence.textContent = `${fusedPct}%`;
        fusedProgressBar.style.width = `${fusedPct}%`;

        // 2. Modality Cards logic
        const mods = data.modalities;
        
        // Audio
        if (mods.audio) {
            cardAudio.classList.remove('inactive');
            audioLabel.textContent = capitalizeFirstLetter(mods.audio.label);
            const audPct = (mods.audio.confidence * 100).toFixed(1);
            audioConf.textContent = `Confidence: ${audPct}%`;
            barAudio.style.width = `${audPct}%`;
        } else {
            cardAudio.classList.add('inactive');
            audioLabel.textContent = "Unavailable";
            audioConf.textContent = "Confidence: N/A";
            barAudio.style.width = "0%";
        }

        // Face / Video
        if (mods.face_video) {
            cardFace.classList.remove('inactive');
            faceLabel.textContent = capitalizeFirstLetter(mods.face_video.label);
            const facePct = (mods.face_video.confidence * 100).toFixed(1);
            faceConf.textContent = `Confidence: ${facePct}%`;
            barFace.style.width = `${facePct}%`;
        } else {
            cardFace.classList.add('inactive');
            faceLabel.textContent = "Unavailable";
            faceConf.textContent = "Confidence: N/A";
            barFace.style.width = "0%";
        }

        // Text
        if (mods.text) {
            cardText.classList.remove('inactive');
            textLabel.textContent = capitalizeFirstLetter(mods.text.label);
            const txtPct = (mods.text.confidence * 100).toFixed(1);
            textConf.textContent = `Confidence: ${txtPct}%`;
            barText.style.width = `${txtPct}%`;
        } else {
            cardText.classList.add('inactive');
            textLabel.textContent = "Unavailable";
            textConf.textContent = "Confidence: N/A";
            barText.style.width = "0%";
        }

        // 3. Render Chart
        renderChart(data);
    }

    function renderChart(data) {
        const labels = ['Angry', 'Disgust', 'Fearful', 'Happy', 'Neutral', 'Sad', 'Surprised'];
        const datasets = [];
        const mods = data.modalities;
        
        if (mods.audio) {
            datasets.push({
                label: 'Audio Modality',
                data: mods.audio.probs,
                backgroundColor: 'rgba(99, 102, 241, 0.25)',
                borderColor: 'rgba(99, 102, 241, 1)',
                borderWidth: 1.5,
                borderRadius: 4
            });
        }
        
        if (mods.face_video) {
            datasets.push({
                label: 'Facial Expression',
                data: mods.face_video.probs,
                backgroundColor: 'rgba(236, 72, 153, 0.25)',
                borderColor: 'rgba(236, 72, 153, 1)',
                borderWidth: 1.5,
                borderRadius: 4
            });
        }
        
        if (mods.text) {
            datasets.push({
                label: 'Text Context',
                data: mods.text.probs,
                backgroundColor: 'rgba(234, 179, 8, 0.25)',
                borderColor: 'rgba(234, 179, 8, 1)',
                borderWidth: 1.5,
                borderRadius: 4
            });
        }
        
        const fuse_active = data.fusion.late_fusion || data.fusion.early_fusion;
        if (fuse_active) {
            datasets.push({
                label: 'Fused Prediction',
                data: fuse_active.probs,
                backgroundColor: 'rgba(168, 85, 247, 0.7)',
                borderColor: 'rgba(168, 85, 247, 1)',
                borderWidth: 2,
                borderRadius: 6
            });
        }

        const ctx = document.getElementById('probsChart').getContext('2d');
        
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
                        grid: { color: 'rgba(255, 255, 255, 0.04)' },
                        ticks: { color: '#9ca3af', font: { family: 'Inter', size: 10 } }
                    },
                    y: {
                        min: 0,
                        max: 1.0,
                        grid: { color: 'rgba(255, 255, 255, 0.04)' },
                        ticks: { color: '#9ca3af', font: { family: 'Inter', size: 10 } }
                    }
                },
                plugins: {
                    legend: {
                        labels: { color: '#f3f4f6', font: { family: 'Inter', weight: 'medium', size: 10 } }
                    },
                    tooltip: {
                        backgroundColor: '#0b0f19',
                        titleFont: { family: 'Outfit', weight: 'bold' },
                        bodyFont: { family: 'Inter' },
                        padding: 10,
                        borderWidth: 1,
                        borderColor: 'rgba(255, 255, 255, 0.06)'
                    }
                }
            }
        });
    }

    // ============================================================
    // LIVE WEBCAM & MICROPHONE CAPTURE IMPLEMENTATION
    // ============================================================

    const btnToggleWebcam = document.getElementById('btn-toggle-webcam');
    const webcamBox = document.getElementById('webcam-box');
    const webcamElement = document.getElementById('webcam-element');
    const btnRecordVideo = document.getElementById('btn-record-video');
    const videoRecordingStatus = document.getElementById('video-recording-status');

    const btnRecordMic = document.getElementById('btn-record-mic');
    const audioRecordingStatus = document.getElementById('audio-recording-status');

    let webcamStream = null;
    let videoRecorder = null;
    let videoChunks = [];
    let isRecordingVideo = false;
    let videoTimerInterval = null;

    let micStream = null;
    let audioRecorder = null;
    let audioChunks = [];
    let isRecordingAudio = false;
    let audioTimerInterval = null;

    // Live Webcam toggle (Video only, facing user)
    btnToggleWebcam.addEventListener('click', async () => {
        if (webcamBox.style.display === 'none') {
            try {
                // Request video only to prevent audio device conflicts
                webcamStream = await navigator.mediaDevices.getUserMedia({
                    video: { facingMode: "user" },
                    audio: false
                });
                webcamElement.srcObject = webcamStream;
                webcamBox.style.display = 'block';
                btnToggleWebcam.innerHTML = '<i class="fa-solid fa-camera-slash"></i> Disable Camera';
                videoRecordingStatus.textContent = "Camera active. Ready to record.";
            } catch (err) {
                alert(`Webcam access failed: ${err.message}. Please verify camera permissions.`);
            }
        } else {
            stopWebcamStream();
        }
    });

    function stopWebcamStream() {
        if (webcamStream) {
            webcamStream.getTracks().forEach(track => track.stop());
            webcamStream = null;
        }
        webcamElement.srcObject = null;
        webcamBox.style.display = 'none';
        btnToggleWebcam.innerHTML = '<i class="fa-solid fa-camera"></i> Live Camera Feed';
        videoRecordingStatus.textContent = "";
        
        if (videoTimerInterval) {
            clearInterval(videoTimerInterval);
            videoTimerInterval = null;
        }
        isRecordingVideo = false;
        btnRecordVideo.innerHTML = '<i class="fa-solid fa-circle"></i> Start Recording';
    }

    // Live Webcam Recording using standard HTML5 MediaRecorder (fully responsive & un-frozen)
    btnRecordVideo.addEventListener('click', () => {
        if (!isRecordingVideo) {
            if (!webcamStream) {
                alert("Please enable the camera feed first.");
                return;
            }
            isRecordingVideo = true;
            videoChunks = [];
            
            // Standard mimeType check for cross-browser support
            let options = {};
            if (MediaRecorder.isTypeSupported('video/webm;codecs=vp9')) {
                options = { mimeType: 'video/webm;codecs=vp9' };
            } else if (MediaRecorder.isTypeSupported('video/webm')) {
                options = { mimeType: 'video/webm' };
            } else if (MediaRecorder.isTypeSupported('video/mp4')) {
                options = { mimeType: 'video/mp4' }; // Safari fallback
            }

            try {
                videoRecorder = new MediaRecorder(webcamStream, options);
            } catch (e) {
                videoRecorder = new MediaRecorder(webcamStream);
            }

            videoRecorder.ondataavailable = (e) => {
                if (e.data && e.data.size > 0) {
                    videoChunks.push(e.data);
                }
            };

            videoRecorder.onstop = () => {
                const mimeType = videoRecorder.mimeType || 'video/webm';
                const blob = new Blob(videoChunks, { type: mimeType });
                const ext = mimeType.includes('mp4') ? 'mp4' : 'webm';
                recordedVideoFile = new File([blob], `webcam_recording.${ext}`, { type: mimeType });
                videoRecordingStatus.textContent = "Recording saved successfully. Ready for inference.";
                btnRecordVideo.innerHTML = '<i class="fa-solid fa-circle"></i> Start Recording';
                isRecordingVideo = false;
            };

            videoRecorder.start();
            btnRecordVideo.innerHTML = '<i class="fa-solid fa-stop"></i> Stop Recording';

            let elapsed = 0;
            videoRecordingStatus.textContent = `Recording: 00:00`;
            videoTimerInterval = setInterval(() => {
                elapsed++;
                const mins = String(Math.floor(elapsed / 60)).padStart(2, '0');
                const secs = String(elapsed % 60).padStart(2, '0');
                videoRecordingStatus.textContent = `Recording: ${mins}:${secs}`;
            }, 1000);
        } else {
            // Stop recording
            if (videoRecorder && videoRecorder.state === "recording") {
                videoRecorder.stop();
            }
            if (videoTimerInterval) {
                clearInterval(videoTimerInterval);
                videoTimerInterval = null;
            }
        }
    });

    // Modern Cross-Browser Microphone Voice Recording using HTML5 MediaRecorder (safely handles webm/mp4 inputs)
    btnRecordMic.addEventListener('click', async () => {
        if (!isRecordingAudio) {
            isRecordingAudio = true;
            audioChunks = [];
            
            try {
                micStream = await navigator.mediaDevices.getUserMedia({ audio: true });
                
                let options = {};
                if (MediaRecorder.isTypeSupported('audio/webm')) {
                    options = { mimeType: 'audio/webm' };
                } else if (MediaRecorder.isTypeSupported('audio/mp4')) {
                    options = { mimeType: 'audio/mp4' }; // Safari fallback
                }

                try {
                    audioRecorder = new MediaRecorder(micStream, options);
                } catch (e) {
                    audioRecorder = new MediaRecorder(micStream);
                }

                audioRecorder.ondataavailable = (e) => {
                    if (e.data && e.data.size > 0) {
                        audioChunks.push(e.data);
                    }
                };

                audioRecorder.onstop = () => {
                    const mimeType = audioRecorder.mimeType || 'audio/webm';
                    const blob = new Blob(audioChunks, { type: mimeType });
                    const ext = mimeType.includes('mp4') ? 'mp4' : 'webm';
                    recordedAudioFile = new File([blob], `mic_recording.${ext}`, { type: mimeType });
                    audioRecordingStatus.textContent = "Voice clip saved. Ready for inference.";
                    btnRecordMic.innerHTML = '<i class="fa-solid fa-microphone"></i> Live Voice Record';
                    isRecordingAudio = false;
                };

                audioRecorder.start();
                btnRecordMic.innerHTML = '<i class="fa-solid fa-stop"></i> Stop Recording';
                audioRecordingStatus.textContent = "Recording voice: 00:00";

                let elapsed = 0;
                audioTimerInterval = setInterval(() => {
                    elapsed++;
                    const mins = String(Math.floor(elapsed / 60)).padStart(2, '0');
                    const secs = String(elapsed % 60).padStart(2, '0');
                    audioRecordingStatus.textContent = `Recording voice: ${mins}:${secs}`;
                }, 1000);

            } catch (err) {
                alert(`Microphone access failed: ${err.message}. Please check permissions.`);
                btnRecordMic.innerHTML = '<i class="fa-solid fa-microphone"></i> Live Voice Record';
                audioRecordingStatus.textContent = "";
                isRecordingAudio = false;
            }
        } else {
            // Stop recording
            isRecordingAudio = false;
            if (audioTimerInterval) {
                clearInterval(audioTimerInterval);
                audioTimerInterval = null;
            }
            if (audioRecorder && audioRecorder.state === "recording") {
                audioRecorder.stop();
            }
            if (micStream) {
                micStream.getTracks().forEach(track => track.stop());
                micStream = null;
            }
        }
    });

    function capitalizeFirstLetter(string) {
        if (!string) return '';
        if (string.toLowerCase() === 'fearful') return 'Fearful';
        if (string.toLowerCase() === 'surprised') return 'Surprised';
        return string.charAt(0).toUpperCase() + string.slice(1);
    }
});
