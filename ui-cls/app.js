// YOLO Classification Vue App
// Handles file uploads, camera controls, and displays results

// Vue 2 app instance
new Vue({
  el: '#app',
  data: {
    // File handling
    selectedFile: null,
    previewImage: null,

    // Classification state
    isClassifying: false,
    predictions: [],

    // Camera state
    cameraRunning: false,
    cameraFeedUrl: '/video_feed',

    // Detection state
    latestDetection: { class: '', confidence: 0, timestamp: 0 },
    detectionHistory: [],

    // Alert state
    showDetectionAlert: false,
    alertDetection: { class: '', confidence: 0 },

    // UI state
    statusMessage: '',
    statusType: 'info', // 'info', 'success', 'error'
    isLoading: false,
    imageStatus: 'Select an image to classify',

    // Model info
    classes: []
  },

  mounted() {
    this.loadClasses();
    console.log('YOLO Classification app started');
  },

  methods: {
    /**
     * Trigger the hidden file input when upload area is clicked
     */
    triggerFileInput() {
      this.$refs.fileInput.click();
    },

    /**
     * Handle file selection from input
     */
    handleFileSelect(event) {
      const file = event.target.files[0];
      if (file) {
        this.setSelectedFile(file);
      }
    },

    /**
     * Handle drag and drop file selection
     */
    handleDrop(event) {
      event.preventDefault();
      const files = event.dataTransfer.files;
      if (files.length > 0) {
        this.setSelectedFile(files[0]);
      }
    },

    /**
     * Set the selected file and create preview
     */
    setSelectedFile(file) {
      // Validate file type
      if (!file.type.startsWith('image/')) {
        this.showStatus('Please select an image file', 'error');
        return;
      }

      this.selectedFile = file;

      // Create preview
      const reader = new FileReader();
      reader.onload = (e) => {
        this.previewImage = e.target.result;
        this.imageStatus = 'Image loaded - click Classify to analyze';
      };
      reader.readAsDataURL(file);

      // Clear previous predictions
      this.predictions = [];
    },

    /**
     * Clear the selected file
     */
    clearFile() {
      this.selectedFile = null;
      this.previewImage = null;
      this.predictions = [];
      this.imageStatus = 'Select an image to classify';
      this.$refs.fileInput.value = '';
    },

    /**
     * Send image to backend for classification
     */
    async classifyImage() {
      if (!this.selectedFile) return;

      this.isClassifying = true;
      this.showStatus('Classifying image...', 'info', true);

      try {
        const formData = new FormData();
        formData.append('image', this.selectedFile);

        const response = await axios.post('/api/classify', formData, {
          headers: {
            'Content-Type': 'multipart/form-data'
          }
        });

        if (response.data.error) {
          throw new Error(response.data.error);
        }

        // Update predictions
        this.predictions = response.data.predictions || [];

        // Update preview image if server sent one back
        if (response.data.image) {
          this.previewImage = response.data.image;
        }

        this.imageStatus = 'Classification complete';
        this.showStatus('Classification complete!', 'success');

      } catch (error) {
        console.error('Classification error:', error);
        const errorMsg = error.response?.data?.error || error.message;
        this.showStatus(`Error: ${errorMsg}`, 'error');
      } finally {
        this.isClassifying = false;
      }
    },



    /**
     * Start live camera classification
     */
    async startCamera() {
      this.showStatus('Starting camera...', 'info', true);

      try {
        const response = await axios.post('/api/start-camera');
        this.cameraRunning = true;
        this.cameraFeedUrl = '/video_feed?' + new Date().getTime(); // Force refresh
        this.showStatus('Camera started!', 'success');

        // Start polling for detections
        this.startDetectionPolling();

      } catch (error) {
        console.error('Camera start error:', error);
        const errorMsg = error.response?.data?.error || error.message;
        this.showStatus(`Error starting camera: ${errorMsg}`, 'error');
      }
    },

    /**
     * Stop live camera classification
     */
    async stopCamera() {
      this.showStatus('Stopping camera...', 'info', true);

      try {
        await axios.post('/api/stop-camera');
        this.cameraRunning = false;
        this.cameraFeedUrl = '/video_feed?' + new Date().getTime(); // Force refresh
        this.showStatus('Camera stopped', 'success');

        // Stop polling for detections
        this.stopDetectionPolling();

      } catch (error) {
        console.error('Camera stop error:', error);
        const errorMsg = error.response?.data?.error || error.message;
        this.showStatus(`Error stopping camera: ${errorMsg}`, 'error');
      }
    },

    /**
     * Start polling for high-confidence detections and camera status
     */
    startDetectionPolling() {
      this.detectionInterval = setInterval(async () => {
        try {
          // Get detections
          const detectionsResponse = await axios.get('/api/detections');
          const newLatest = detectionsResponse.data.latest;

          // Check if there's a new detection to show alert for
          if (newLatest.class && newLatest.timestamp !== this.latestDetection.timestamp) {
            this.showDetectionAlert = true;
            this.alertDetection = {
              class: newLatest.class,
              confidence: newLatest.confidence
            };

            // Auto-dismiss alert after 5 seconds
            setTimeout(() => {
              this.showDetectionAlert = false;
            }, 5000);
          }

          this.latestDetection = newLatest;
          this.detectionHistory = detectionsResponse.data.history;

          // Check camera status (auto-stop detection)
          const statusResponse = await axios.get('/api/camera-status');
          if (!statusResponse.data.running && this.cameraRunning) {
            // Camera was auto-stopped due to high confidence detection
            this.cameraRunning = false;
            this.cameraFeedUrl = '/video_feed?' + new Date().getTime();
            this.showStatus('Camera auto-stopped after high confidence detection!', 'success');
            this.stopDetectionPolling();
          }
        } catch (error) {
          console.error('Error fetching detections:', error);
        }
      }, 1000); // Poll every second
    },

    /**
     * Stop polling for detections
     */
    stopDetectionPolling() {
      if (this.detectionInterval) {
        clearInterval(this.detectionInterval);
        this.detectionInterval = null;
      }
    },

    /**
     * Clear detection history
     */
    async clearDetections() {
      try {
        await axios.post('/api/clear-detections');
        this.latestDetection = { class: '', confidence: 0, timestamp: 0 };
        this.detectionHistory = [];
        this.showStatus('Detection history cleared', 'success');
      } catch (error) {
        console.error('Error clearing detections:', error);
        this.showStatus('Error clearing detections', 'error');
      }
    },

    /**
     * Format timestamp for display
     */
    formatTime(timestamp) {
      if (!timestamp) return '';
      const date = new Date(timestamp * 1000);
      return date.toLocaleTimeString();
    },

    /**
     * Dismiss the detection alert
     */
    dismissAlert() {
      this.showDetectionAlert = false;
    },

    /**
     * Load available classes from the backend
     */
    async loadClasses() {
      try {
        const response = await axios.get('/api/classes');
        this.classes = response.data.classes || [];
      } catch (error) {
        console.error('Error loading classes:', error);
        this.classes = [];
      }
    },

    /**
     * Show status message to user
     */
    showStatus(message, type = 'info', loading = false) {
      this.statusMessage = message;
      this.statusType = type;
      this.isLoading = loading;

      // Auto-hide non-loading messages
      if (!loading) {
        setTimeout(() => {
          this.statusMessage = '';
        }, 3000);
      }
    }
  }
});