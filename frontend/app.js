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

    // Live Webcam toggle
    btnToggleWebcam.addEventListener('click', async () => {
        if (webcamBox.style.display === 'none') {
            try {
                webcamStream = await navigator.mediaDevices.getUserMedia({ video: true, audio: true });
                webcamElement.srcObject = webcamStream;
                webcamBox.style.display = 'block';
                btnToggleWebcam.innerHTML = '<i class="fa-solid fa-camera-slash"></i> Disable Webcam';
                videoRecordingStatus.textContent = "Webcam active. Ready to record.";
            } catch (err) {
                alert(`Webcam access failed: ${err.message}`);
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
        btnToggleWebcam.innerHTML = '<i class="fa-solid fa-camera"></i> Use Live Webcam';
        videoRecordingStatus.textContent = "";
    }

    // Live Webcam Recording
    let videoTimerInterval = null;
    let isRecordingVideo = false;

    btnRecordVideo.addEventListener('click', () => {
        if (!isRecordingVideo) {
            if (!webcamStream) {
                alert("Please enable the webcam first.");
                return;
            }
            isRecordingVideo = true;
            videoChunks = [];
            
            try {
                videoRecorder = new MediaRecorder(webcamStream);
            } catch (e) {
                videoRecorder = new MediaRecorder(webcamStream); // default fallback
            }

            videoRecorder.ondataavailable = (e) => {
                if (e.data.size > 0) {
                    videoChunks.push(e.data);
                }
            };

            videoRecorder.onstop = () => {
                const blob = new Blob(videoChunks, { type: 'video/webm' });
                recordedVideoFile = new File([blob], "webcam_recording.webm", { type: 'video/webm' });
                videoRecordingStatus.textContent = "Recording complete! Ready for analysis.";
                btnRecordVideo.innerHTML = '<i class="fa-solid fa-circle"></i> Start Recording';
                isRecordingVideo = false;
            };

            videoRecorder.start();
            btnRecordVideo.innerHTML = '<i class="fa-solid fa-stop"></i> Stop Recording';

            let elapsed = 0;
            videoRecordingStatus.textContent = `Recording live feed... 00:00`;
            videoTimerInterval = setInterval(() => {
                elapsed++;
                const mins = String(Math.floor(elapsed / 60)).padStart(2, '0');
                const secs = String(elapsed % 60).padStart(2, '0');
                videoRecordingStatus.textContent = `Recording live feed... ${mins}:${secs}`;
            }, 1000);
        } else {
            // Stop video recording
            if (videoRecorder && videoRecorder.state === "recording") {
                videoRecorder.stop();
            }
            if (videoTimerInterval) {
                clearInterval(videoTimerInterval);
                videoTimerInterval = null;
            }
        }
    });

    // Live Microphone WAV Recording
    let micStream = null;
    let audioContext = null;
    let audioProcessor = null;
    let leftChannel = [];
    let recordingLength = 0;
    let sampleRate = 44100;
    let audioTimerInterval = null;
    let isRecordingAudio = false;

    btnRecordMic.addEventListener('click', async () => {
        if (!isRecordingAudio) {
            isRecordingAudio = true;
            leftChannel = [];
            recordingLength = 0;
            btnRecordMic.innerHTML = '<i class="fa-solid fa-stop"></i> Stop Recording';
            audioRecordingStatus.textContent = "Recording voice... 00:00";

            try {
                micStream = await navigator.mediaDevices.getUserMedia({ audio: true });
                audioContext = new (window.AudioContext || window.webkitAudioContext)();
                sampleRate = audioContext.sampleRate;
                
                const source = audioContext.createMediaStreamSource(micStream);
                audioProcessor = audioContext.createScriptProcessor(4096, 1, 1);
                
                audioProcessor.onaudioprocess = function(e) {
                    const left = e.inputBuffer.getChannelData(0);
                    leftChannel.push(new Float32Array(left));
                    recordingLength += 4096;
                };
                
                source.connect(audioProcessor);
                audioProcessor.connect(audioContext.destination);

                let elapsed = 0;
                audioTimerInterval = setInterval(() => {
                    elapsed++;
                    const mins = String(Math.floor(elapsed / 60)).padStart(2, '0');
                    const secs = String(elapsed % 60).padStart(2, '0');
                    audioRecordingStatus.textContent = `Recording voice... ${mins}:${secs}`;
                }, 1000);

            } catch (err) {
                alert(`Microphone access failed: ${err.message}`);
                btnRecordMic.innerHTML = '<i class="fa-solid fa-microphone"></i> Record Speech';
                audioRecordingStatus.textContent = "";
                isRecordingAudio = false;
            }
        } else {
            isRecordingAudio = false;
            if (audioTimerInterval) {
                clearInterval(audioTimerInterval);
                audioTimerInterval = null;
            }
            
            if (audioProcessor) {
                audioProcessor.disconnect();
                if (micStream) {
                    micStream.getTracks().forEach(track => track.stop());
                }
            }
            
            // Process buffer and save as WAV file
            const leftBuffer = flattenArray(leftChannel, recordingLength);
            const wavBuffer = writeWavFile(leftBuffer);
            const blob = new Blob([wavBuffer], { type: 'audio/wav' });
            
            recordedAudioFile = new File([blob], "mic_recording.wav", { type: 'audio/wav' });
            
            audioRecordingStatus.textContent = "Recording complete! Ready for analysis.";
            btnRecordMic.innerHTML = '<i class="fa-solid fa-microphone"></i> Record Speech';
        }
    });

    function flattenArray(channelBuffer, recordingLength) {
        const result = new Float32Array(recordingLength);
        let offset = 0;
        for (let i = 0; i < channelBuffer.length; i++) {
            const buffer = channelBuffer[i];
            result.set(buffer, offset);
            offset += buffer.length;
        }
        return result;
    }

    function writeWavFile(buffer) {
        const bufferLength = buffer.length;
        const arrayBuffer = new ArrayBuffer(44 + bufferLength * 2);
        const view = new DataView(arrayBuffer);
        
        writeString(view, 0, 'RIFF');
        view.setUint32(4, 36 + bufferLength * 2, true);
        writeString(view, 8, 'WAVE');
        writeString(view, 12, 'fmt ');
        view.setUint32(16, 16, true);
        view.setUint16(20, 1, true);
        view.setUint16(22, 1, true);
        view.setUint32(24, sampleRate, true);
        view.setUint32(28, sampleRate * 2, true);
        view.setUint16(32, 2, true);
        view.setUint16(34, 16, true);
        writeString(view, 36, 'data');
        view.setUint32(40, bufferLength * 2, true);
        
        let index = 44;
        for (let i = 0; i < bufferLength; i++) {
            let sample = buffer[i];
            if (sample > 1) sample = 1;
            else if (sample < -1) sample = -1;
            view.setInt16(index, sample < 0 ? sample * 0x8000 : sample * 0x7FFF, true);
            index += 2;
        }
        
        return arrayBuffer;
    }

    function writeString(view, offset, string) {
        for (let i = 0; i < string.length; i++) {
            view.setUint8(offset + i, string.charCodeAt(i));
        }
    }

    function capitalizeFirstLetter(string) {
        if (!string) return '';
        if (string.toLowerCase() === 'fearful') return 'Fearful';
        if (string.toLowerCase() === 'surprised') return 'Surprised';
        return string.charAt(0).toUpperCase() + string.slice(1);
    }
});
