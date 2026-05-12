let socket = null;
let mediaRecorder = null;
let targetLanguage = "hi"; // Default

// 1. Initialize WebSocket Connection
const connectBase = () => {
    socket = new WebSocket("ws://localhost:8000/ws/stream");
    socket.binaryType = "arraybuffer";

    socket.onopen = () => {
        // Send initial language config
        socket.send(JSON.stringify({ type: "config", lang: targetLanguage }));
    };

    socket.onmessage = (event) => {
        // Here we receive the cloned audio bytes back from Python
        playByteArray(event.data);
    };
};

// 2. Capture Audio from the YouTube Video element
const startCapture = async () => {
    const videoElement = document.querySelector('video');
    if (!videoElement) return;

    // Capture the stream from the video element
    const stream = videoElement.captureStream ? videoElement.captureStream() : videoElement.mozCaptureStream();
    const audioTrack = stream.getAudioTracks()[0];
    const audioStream = new MediaStream([audioTrack]);

    mediaRecorder = new MediaRecorder(audioStream);

    mediaRecorder.ondataavailable = (event) => {
        if (event.data.size > 0 && socket.readyState === WebSocket.OPEN) {
            // Send the chunk to FastAPI
            socket.send(event.data);
        }
    };

    // Slice audio every 5000ms (5 seconds)
    mediaRecorder.start(5000);
    videoElement.muted = true; // Mute original so we only hear the clone
};

// 3. Playback received chunks using Web Audio API
const audioCtx = new (window.AudioContext || window.webkitAudioContext)();
function playByteArray(bytes) {
    audioCtx.decodeAudioData(bytes, (buffer) => {
        const source = audioCtx.createBufferSource();
        source.buffer = buffer;
        source.connect(audioCtx.destination);
        source.start(0);
    });
}

connectBase();

chrome.runtime.onMessage.addListener((request, sender, sendResponse) => {
    if (request.type === "CHANGE_LANGUAGE") {
        targetLanguage = request.lang;

        const video = document.querySelector('video');
        video.pause();

        // Tell backend to switch language context
        socket.send(JSON.stringify({ type: "config", lang: targetLanguage }));

        // Resume video after a small buffer delay (e.g., 2 seconds)
        setTimeout(() => video.play(), 2000);
    }
});

// Wait for the video element to be available before starting capture
const waitForVideo = setInterval(() => {
    const video = document.querySelector('video');
    if (video && !mediaRecorder) {
        clearInterval(waitForVideo);
        startCapture();
    }
}, 1000);