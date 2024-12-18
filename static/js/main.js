// Main JavaScript file for the Violence Detection System

document.addEventListener('DOMContentLoaded', function() {
    // Initialize tooltips
    var tooltipTriggerList = [].slice.call(document.querySelectorAll('[data-bs-toggle="tooltip"]'));
    var tooltipList = tooltipTriggerList.map(function (tooltipTriggerEl) {
        return new bootstrap.Tooltip(tooltipTriggerEl);
    });

    // Smooth scrolling for navigation links
    document.querySelectorAll('a[href^="#"]').forEach(anchor => {
        anchor.addEventListener('click', function (e) {
            e.preventDefault();
            document.querySelector(this.getAttribute('href')).scrollIntoView({
                behavior: 'smooth'
            });
        });
    });
});

// Function to handle starting the camera feed
async function startCamera() {
    const video = document.getElementById('videoElement');
    const startButton = document.getElementById('startDetection');

    try {
        const stream = await navigator.mediaDevices.getUserMedia({ video: true });
        video.srcObject = stream;
        video.style.display = 'block';
        startButton.innerHTML = '<i class="fas fa-stop-circle me-2"></i>Stop Detection';
        startButton.onclick = stopCamera;
    } catch (err) {
        console.error('Error accessing camera:', err);
        alert('Error accessing camera. Please make sure you have granted camera permissions.');
    }
}

function stopCamera() {
    const video = document.getElementById('videoElement');
    const startButton = document.getElementById('startDetection');
    
    const stream = video.srcObject;
    const tracks = stream.getTracks();
    tracks.forEach(track => track.stop());
    
    video.srcObject = null;
    video.style.display = 'none';
    startButton.innerHTML = '<i class="fas fa-play-circle me-2"></i>Start Detection';
    startButton.onclick = startCamera;
}

// Function to handle alerts (can be called when violence is detected)
function showAlert(message) {
    const alertHtml = `
        <div class="alert alert-danger alert-dismissible fade show position-fixed top-0 start-50 translate-middle-x mt-3" role="alert">
            <strong><i class="fas fa-exclamation-triangle me-2"></i>Alert!</strong> ${message}
            <button type="button" class="btn-close" data-bs-dismiss="alert" aria-label="Close"></button>
        </div>
    `;
    
    document.body.insertAdjacentHTML('afterbegin', alertHtml);
}