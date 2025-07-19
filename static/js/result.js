// DOM Elements
const resultImage = document.querySelector('.result-image img');
const confidenceBars = document.querySelectorAll('.confidence-bar, .prediction-bar');

// Initialize
document.addEventListener('DOMContentLoaded', () => {
    // Setup animations
    setupAnimations();
    
    // Setup image zoom functionality
    setupImageZoom();
});

// Setup animations for confidence bars
function setupAnimations() {
    // Animate confidence bars on page load
    confidenceBars.forEach(bar => {
        const width = bar.style.width;
        bar.style.width = '0';
        
        setTimeout(() => {
            bar.style.transition = 'width 1s ease-in-out';
            bar.style.width = width;
        }, 300);
    });
}

// Setup image zoom functionality
function setupImageZoom() {
    if (!resultImage) return;
    
    // Add zoom functionality
    let isZoomed = false;
    
    resultImage.addEventListener('click', () => {
        if (isZoomed) {
            // Zoom out
            resultImage.style.transform = 'scale(1)';
            resultImage.style.cursor = 'zoom-in';
        } else {
            // Zoom in
            resultImage.style.transform = 'scale(1.5)';
            resultImage.style.cursor = 'zoom-out';
        }
        
        isZoomed = !isZoomed;
    });
    
    // Add hover hint
    resultImage.title = 'Click to zoom';
    resultImage.style.cursor = 'zoom-in';
    
    // Add styles for zoom transition
    resultImage.style.transition = 'transform 0.3s ease';
    resultImage.style.transformOrigin = 'center center';
}

// Add custom styling
document.addEventListener('DOMContentLoaded', () => {
    const style = document.createElement('style');
    style.textContent = `
        .result-image img {
            transition: transform 0.3s ease;
            cursor: zoom-in;
        }
        
        .confidence-bar, .prediction-bar {
            transition: width 1s ease-in-out;
        }
        
        @media print {
            header, footer, .back-btn, .additional-info {
                display: none;
            }
            
            .result-container {
                display: block;
                page-break-inside: avoid;
            }
            
            body {
                background-color: white;
                color: black;
            }
        }
    `;
    document.head.appendChild(style);
});