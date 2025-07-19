// DOM Elements
const fileInput = document.getElementById('file-input');
const dropArea = document.getElementById('drop-area');
const uploadContent = document.querySelector('.upload-content');
const previewContainer = document.getElementById('preview-container');
const previewImage = document.getElementById('preview-image');
const removeImageBtn = document.getElementById('remove-image');
const submitBtn = document.getElementById('submit-btn');
const uploadForm = document.getElementById('upload-form');

// Initialize
document.addEventListener('DOMContentLoaded', () => {
    // Setup event listeners
    setupFileUpload();
    setupFormSubmission();
});

// Setup file upload functionality
function setupFileUpload() {
    if (!fileInput || !dropArea) return;

    // File input change event
    fileInput.addEventListener('change', handleFileSelect);

    // Drag and drop events
    ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
        dropArea.addEventListener(eventName, preventDefaults, false);
    });

    function preventDefaults(e) {
        e.preventDefault();
        e.stopPropagation();
    }

    // Highlight drop area when dragging over it
    ['dragenter', 'dragover'].forEach(eventName => {
        dropArea.addEventListener(eventName, highlight, false);
    });

    ['dragleave', 'drop'].forEach(eventName => {
        dropArea.addEventListener(eventName, unhighlight, false);
    });

    function highlight() {
        dropArea.classList.add('highlight');
    }

    function unhighlight() {
        dropArea.classList.remove('highlight');
    }

    // Handle dropped files
    dropArea.addEventListener('drop', handleDrop, false);

    function handleDrop(e) {
        const dt = e.dataTransfer;
        const files = dt.files;
        
        if (files.length) {
            fileInput.files = files;
            handleFileSelect();
        }
    }

    // Remove image button
    if (removeImageBtn) {
        removeImageBtn.addEventListener('click', removeImage);
    }
}

// Handle file selection
function handleFileSelect() {
    if (!fileInput.files || fileInput.files.length === 0) return;
    
    const file = fileInput.files[0];
    
    // Check if file is an image
    if (!file.type.match('image.*')) {
        alert('Please select an image file (jpg, jpeg, or png)');
        removeImage();
        return;
    }
    
    // Check file size (max 10MB)
    if (file.size > 10 * 1024 * 1024) {
        alert('File size too large. Please select an image less than 10MB.');
        removeImage();
        return;
    }
    
    // Display preview
    const reader = new FileReader();
    
    reader.onload = function(e) {
        previewImage.src = e.target.result;
        uploadContent.style.display = 'none';
        previewContainer.style.display = 'block';
        submitBtn.disabled = false;
    };
    
    reader.readAsDataURL(file);
}

// Remove selected image
function removeImage() {
    fileInput.value = '';
    previewImage.src = '';
    uploadContent.style.display = 'block';
    previewContainer.style.display = 'none';
    submitBtn.disabled = true;
}

// Setup form submission
function setupFormSubmission() {
    if (!uploadForm) return;
    
    uploadForm.addEventListener('submit', function(e) {
        // Show loading state
        if (submitBtn) {
            submitBtn.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Processing...';
            submitBtn.disabled = true;
        }
    });
}

// Add custom styling for drag and drop
document.addEventListener('DOMContentLoaded', () => {
    const style = document.createElement('style');
    style.textContent = `
        .file-upload-area.highlight {
            border-color: var(--primary-color);
            background-color: rgba(46, 125, 50, 0.1);
        }
        
        .upload-content {
            transition: all 0.3s ease;
        }
        
        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }
        
        .fa-spinner {
            animation: spin 1s linear infinite;
        }
    `;
    document.head.appendChild(style);
});