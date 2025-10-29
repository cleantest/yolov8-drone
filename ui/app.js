// Simple Vue app for object detection
new Vue({
    el: '#app',
    data: {
        activeTab: 'upload',
        selectedImage: null,
        selectedFile: null,
        isProcessing: false,
        isDragOver: false,
        detectionResult: null,
        isCameraRunning: false,
        cameraStatus: 'Camera stopped'
    },

    methods: {
        // Handle file selection
        handleFileSelect(event) {
            const file = event.target.files[0];
            if (file && file.type.startsWith('image/')) {
                this.selectedFile = file;
                // Show preview
                const reader = new FileReader();
                reader.onload = (e) => this.selectedImage = e.target.result;
                reader.readAsDataURL(file);
                this.detectionResult = null;
            }
        },

        // Handle drag and drop
        handleDrop(event) {
            event.preventDefault();
            this.isDragOver = false;
            const file = event.dataTransfer.files[0];
            if (file && file.type.startsWith('image/')) {
                this.selectedFile = file;
                const reader = new FileReader();
                reader.onload = (e) => this.selectedImage = e.target.result;
                reader.readAsDataURL(file);
                this.detectionResult = null;
            }
        },

        // Send image for detection
        async detectObjects() {
            if (!this.selectedFile) return;
            this.isProcessing = true;

            try {
                const formData = new FormData();
                formData.append('image', this.selectedFile);

                const response = await fetch('/api/detect', {
                    method: 'POST',
                    body: formData
                });

                this.detectionResult = await response.json();
            } catch (error) {
                alert('Detection failed. Make sure the server is running.');
            }

            this.isProcessing = false;
        },

        // Start camera
        async startCamera() {
            this.cameraStatus = 'Starting camera...';
            try {
                const response = await fetch('/api/start-camera', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({})
                });
                const data = await response.json();
                this.isCameraRunning = true;
                this.cameraStatus = data.status + ' - Check OpenCV window';
            } catch (error) {
                this.cameraStatus = 'Failed to start camera';
            }
        },

        // Stop camera
        async stopCamera() {
            this.cameraStatus = 'Stopping camera...';
            try {
                const response = await fetch('/api/stop-camera', { method: 'POST' });
                const data = await response.json();
                this.isCameraRunning = false;
                this.cameraStatus = data.status;
            } catch (error) {
                this.cameraStatus = 'Error stopping camera';
            }
        }
    }
});