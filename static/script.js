/**
 * Maverick AI - Frontend JavaScript
 * Professional, robust client-side application with WebSocket support
 * 
 * Features:
 * - Real-time WebSocket communication
 * - Responsive design with smooth animations
 * - Session management and conversation history
 * - Theme switching and user preferences
 * - Auto-resizing textarea and typing indicators
 * - Error handling and connection status
 * - Mobile-first responsive design
 * 
 * @author Maverick AI Team
 * @version 2.1.0
 */

class MaverickAI {
    constructor() {
        // Application state
        this.state = {
            isConnected: false,
            isTyping: false,
            currentSession: null,
            messageCount: 0,
            searchCount: 0,
            theme: 'dark',
            isMobile: window.innerWidth <= 768,
            websocket: null,
            connectionAttempts: 0,
            maxReconnectAttempts: 5
        };

        // Configuration
        this.config = {
            websocketUrl: `${window.location.protocol === 'https:' ? 'wss:' : 'ws:'}//${window.location.host}/ws/`,
            apiUrl: '/api',
            maxMessageLength: 2000,
            reconnectDelay: 1000,
            typingTimeout: 3000,
            animationDuration: 300
        };

        // DOM elements cache
        this.elements = {};
        
        // Bind methods
        this.handleResize = this.debounce(this.handleResize.bind(this), 250);
        this.handleInput = this.debounce(this.handleInput.bind(this), 100);
        
        // Initialize application
        this.init();
    }

    /**
     * Initialize the application
     */
    async init() {
        try {
            await this.cacheElements();
            this.setupEventListeners();
            this.initializeUI();
            this.loadUserPreferences();
            this.generateSessionId();
            await this.connectWebSocket();
            this.hideLoadingScreen();
            
            console.log('🚀 Maverick AI initialized successfully');
        } catch (error) {
            console.error('❌ Failed to initialize Maverick AI:', error);
            this.showError('Failed to initialize application');
        }
    }

    /**
     * Cache DOM elements for performance
     */
    async cacheElements() {
        const elementSelectors = {
            // Main elements
            body: 'body',
            app: '.app',
            sidebar: '#sidebar',
            sidebarBackdrop: '#sidebarBackdrop',
            chatMessages: '#chatMessages',
            welcomeScreen: '#welcomeScreen',
            messageInput: '#messageInput',
            sendBtn: '#sendBtn',
            
            // Header elements
            sidebarToggle: '#sidebarToggle',
            sidebarClose: '#sidebarClose',
            connectionStatus: '#connectionStatus',
            connectionText: '#connectionText',
            messageCount: '#messageCount',
            searchCount: '#searchCount',
            themeToggle: '#themeToggle',
            settingsBtn: '#settingsBtn',
            headerNewSessionBtn: '#headerNewSessionBtn',
            
            // Input area
            attachBtn: '#attachBtn',
            voiceBtn: '#voiceBtn',
            charCount: '#charCount',
            
            // Sidebar elements
            sessionList: '#sessionList',
            searchSessions: '#searchSessions',
            newSessionBtn: '#newSessionBtn',
            exportBtn: '#exportBtn',
            clearAllBtn: '#clearAllBtn',
            
            // Other elements
            loadingScreen: '#loadingScreen',
            mobileOverlay: '#mobileOverlay',
            messageTemplate: '#message-template'
        };

        for (const [key, selector] of Object.entries(elementSelectors)) {
            const element = document.querySelector(selector);
            if (element) {
                this.elements[key] = element;
            } else {
                console.warn(`Element not found: ${selector}`);
            }
        }

        // Cache suggestion chips
        this.elements.suggestionChips = document.querySelectorAll('.suggestion-chip');
        this.elements.filterBtns = document.querySelectorAll('.filter-btn');
    }

    /**
     * Setup all event listeners
     */
    setupEventListeners() {
        // Window events
        window.addEventListener('resize', this.handleResize);
        window.addEventListener('beforeunload', this.handleBeforeUnload.bind(this));
        window.addEventListener('online', this.handleOnline.bind(this));
        window.addEventListener('offline', this.handleOffline.bind(this));

        // Sidebar events
        this.elements.sidebarToggle?.addEventListener('click', this.toggleSidebar.bind(this));
        this.elements.sidebarClose?.addEventListener('click', this.closeSidebar.bind(this));
        this.elements.sidebarBackdrop?.addEventListener('click', this.closeSidebar.bind(this));

        // Input events
        this.elements.messageInput?.addEventListener('input', this.handleInput.bind(this));
        this.elements.messageInput?.addEventListener('keydown', this.handleKeydown.bind(this));
        this.elements.messageInput?.addEventListener('paste', this.handlePaste.bind(this));
        this.elements.sendBtn?.addEventListener('click', this.sendMessage.bind(this));

        // Theme and settings
        this.elements.themeToggle?.addEventListener('click', this.toggleTheme.bind(this));
        this.elements.settingsBtn?.addEventListener('click', this.openSettings.bind(this));

        // Suggestion chips
        this.elements.suggestionChips?.forEach(chip => {
            chip.addEventListener('click', this.handleSuggestionClick.bind(this));
        });

        // Session management
        this.elements.newSessionBtn?.addEventListener('click', this.createNewSession.bind(this));
        this.elements.exportBtn?.addEventListener('click', this.exportConversation.bind(this));
        this.elements.clearAllBtn?.addEventListener('click', this.clearAllSessions.bind(this));
        this.elements.headerNewSessionBtn?.addEventListener('click', this.createNewSession.bind(this));

        // Filter buttons
        this.elements.filterBtns?.forEach(btn => {
            btn.addEventListener('click', this.handleFilterClick.bind(this));
        });

        // Search sessions
        this.elements.searchSessions?.addEventListener('input', this.handleSessionSearch.bind(this));

        // Voice and attachment buttons
        this.elements.voiceBtn?.addEventListener('click', this.handleVoiceInput.bind(this));
        this.elements.attachBtn?.addEventListener('click', this.handleFileAttachment.bind(this));

        // Mobile overlay
        this.elements.mobileOverlay?.addEventListener('click', this.closeSidebar.bind(this));
    }

    /**
     * Initialize UI components and states
     */
    initializeUI() {
        // Initialize theme
        this.applyTheme(this.state.theme);
        
        // Initialize Lucide icons
        if (typeof lucide !== 'undefined') {
            lucide.createIcons();
        }

        // Setup mobile detection
        this.handleResize();

        // Initialize input area
        this.updateCharCount();
        this.updateSendButton();

        // Setup connection status
        this.updateConnectionStatus(false);

        // Initialize stats
        this.updateStats();
    }

    /**
     * Generate unique session ID
     */
    generateSessionId() {
        this.state.currentSession = 'session_' + Date.now() + '_' + Math.random().toString(36).substr(2, 9);
        console.log('Generated session ID:', this.state.currentSession);
    }

    /**
     * Connect to WebSocket server
     */
    async connectWebSocket() {
        if (this.state.websocket && this.state.websocket.readyState === WebSocket.OPEN) {
            return;
        }

        try {
            const wsUrl = this.config.websocketUrl + this.state.currentSession;
            console.log('Connecting to WebSocket:', wsUrl);
            
            this.state.websocket = new WebSocket(wsUrl);
            
            this.state.websocket.onopen = this.handleWebSocketOpen.bind(this);
            this.state.websocket.onclose = this.handleWebSocketClose.bind(this);
            this.state.websocket.onerror = this.handleWebSocketError.bind(this);
            this.state.websocket.onmessage = this.handleWebSocketMessage.bind(this);
            
        } catch (error) {
            console.error('WebSocket connection failed:', error);
            this.handleConnectionError();
        }
    }

    /**
     * WebSocket event handlers
     */
    handleWebSocketOpen() {
        console.log('✅ WebSocket connected');
        this.state.isConnected = true;
        this.state.connectionAttempts = 0;
        this.updateConnectionStatus(true);
    }

    handleWebSocketClose(event) {
        console.log('🔌 WebSocket disconnected:', event.code, event.reason);
        this.state.isConnected = false;
        this.updateConnectionStatus(false);
        
        // Attempt reconnection if not intentional close
        if (event.code !== 1000 && this.state.connectionAttempts < this.config.maxReconnectAttempts) {
            this.attemptReconnection();
        }
    }

    handleWebSocketError(error) {
        console.error('❌ WebSocket error:', error);
        this.handleConnectionError();
    }

    handleWebSocketMessage(event) {
        try {
            const data = JSON.parse(event.data);
            
            switch (data.type) {
                case 'message':
                    this.displayAIMessage(data.message);
                    this.hideTypingIndicator();
                    break;
                    
                case 'typing':
                    this.showTypingIndicator();
                    break;
                    
                case 'error':
                    this.showError(data.message);
                    this.hideTypingIndicator();
                    break;
                    
                default:
                    console.warn('Unknown message type:', data.type);
            }
        } catch (error) {
            console.error('Error parsing WebSocket message:', error);
        }
    }

    /**
     * Attempt to reconnect WebSocket
     */
    async attemptReconnection() {
        this.state.connectionAttempts++;
        const delay = this.config.reconnectDelay * Math.pow(2, this.state.connectionAttempts - 1);
        
        console.log(`Attempting reconnection ${this.state.connectionAttempts}/${this.config.maxReconnectAttempts} in ${delay}ms`);
        
        setTimeout(() => {
            this.connectWebSocket();
        }, delay);
    }

    /**
     * Handle connection errors
     */
    handleConnectionError() {
        this.state.isConnected = false;
        this.updateConnectionStatus(false);
        this.showError('Connection lost. Attempting to reconnect...');
    }

    /**
     * Send message to AI
     */
    async sendMessage() {
        const message = this.elements.messageInput?.value.trim();
        
        if (!message || message.length === 0) {
            this.focusInput();
            return;
        }

        if (message.length > this.config.maxMessageLength) {
            this.showError(`Message too long. Maximum ${this.config.maxMessageLength} characters allowed.`);
            return;
        }

        if (!this.state.isConnected) {
            this.showError('Not connected to server. Please wait for reconnection.');
            return;
        }

        try {
            // Display user message
            this.displayUserMessage(message);
            
            // Clear input and hide welcome screen
            this.elements.messageInput.value = '';
            this.updateCharCount();
            this.updateSendButton();
            this.hideWelcomeScreen();
            
            // Send via WebSocket
            this.state.websocket.send(JSON.stringify({
                message: message,
                timestamp: new Date().toISOString()
            }));
            
            // Update stats
            this.state.messageCount++;
            this.updateStats();
            
            // Auto-resize textarea
            this.autoResizeTextarea();
            
        } catch (error) {
            console.error('Error sending message:', error);
            this.showError('Failed to send message. Please try again.');
        }
    }

    /**
     * Display user message in chat
     */
    displayUserMessage(message) {
        const messageElement = this.createMessageElement({
            content: message,
            isUser: true,
            timestamp: new Date()
        });
        
        this.appendMessage(messageElement);
        this.scrollToBottom();
    }

    /**
     * Display AI message in chat
     */
    displayAIMessage(message) {
        const messageElement = this.createMessageElement({
            content: message,
            isUser: false,
            timestamp: new Date()
        });
        
        this.appendMessage(messageElement);
        this.scrollToBottom();
    }

    /**
     * Create message DOM element
     */
    createMessageElement({ content, isUser, timestamp }) {
        const messageDiv = document.createElement('div');
        messageDiv.className = `message ${isUser ? 'user-message' : 'ai-message'}`;
        
        const avatar = document.createElement('div');
        avatar.className = 'message-avatar';
        avatar.innerHTML = isUser ? 
            '<i data-lucide="user"></i>' : 
            '<i data-lucide="brain-circuit"></i>';
        
        const messageContent = document.createElement('div');
        messageContent.className = 'message-content';
        
        const messageText = document.createElement('div');
        messageText.className = 'message-text';
        
        // Process message content (markdown support)
        if (!isUser && typeof marked !== 'undefined') {
            messageText.innerHTML = marked.parse(content);
            
            // Highlight code blocks
            if (typeof hljs !== 'undefined') {
                messageText.querySelectorAll('pre code').forEach(block => {
                    hljs.highlightElement(block);
                });
            }
        } else {
            messageText.textContent = content;
        }
        
        const messageTimestamp = document.createElement('div');
        messageTimestamp.className = 'message-timestamp';
        messageTimestamp.textContent = this.formatTimestamp(timestamp);
        
        messageContent.appendChild(messageText);
        messageContent.appendChild(messageTimestamp);
        
        messageDiv.appendChild(avatar);
        messageDiv.appendChild(messageContent);
        
        // Add animation
        messageDiv.style.opacity = '0';
        messageDiv.style.transform = 'translateY(20px)';
        
        return messageDiv;
    }

    /**
     * Append message to chat with animation
     */
    appendMessage(messageElement) {
        this.elements.chatMessages?.appendChild(messageElement);
        
        // Refresh icons
        if (typeof lucide !== 'undefined') {
            lucide.createIcons();
        }
        
        // Animate in
        setTimeout(() => {
            messageElement.style.transition = `opacity ${this.config.animationDuration}ms ease, transform ${this.config.animationDuration}ms ease`;
            messageElement.style.opacity = '1';
            messageElement.style.transform = 'translateY(0)';
        }, 50);
    }

    /**
     * Show/hide typing indicator
     */
    showTypingIndicator() {
        if (this.state.isTyping) return;
        
        this.state.isTyping = true;
        
        const typingDiv = document.createElement('div');
        typingDiv.className = 'message ai-message typing-indicator';
        typingDiv.id = 'typing-indicator';
        
        typingDiv.innerHTML = `
            <div class="message-avatar">
                <i data-lucide="brain-circuit"></i>
            </div>
            <div class="message-content">
                <div class="typing-animation">
                    <div class="typing-dot"></div>
                    <div class="typing-dot"></div>
                    <div class="typing-dot"></div>
                </div>
            </div>
        `;
        
        this.elements.chatMessages?.appendChild(typingDiv);
        if (typeof lucide !== 'undefined') {
            lucide.createIcons();
        }
        this.scrollToBottom();
        
        // Auto-hide after timeout
        setTimeout(() => {
            this.hideTypingIndicator();
        }, this.config.typingTimeout);
    }

    hideTypingIndicator() {
        const typingIndicator = document.getElementById('typing-indicator');
        if (typingIndicator) {
            typingIndicator.remove();
        }
        this.state.isTyping = false;
    }

    /**
     * Input handling
     */
    handleInput(event) {
        this.updateCharCount();
        this.updateSendButton();
        this.autoResizeTextarea();
    }

    handleKeydown(event) {
        // Send on Enter (without Shift)
        if (event.key === 'Enter' && !event.shiftKey) {
            event.preventDefault();
            this.sendMessage();
        }
        
        // Auto-resize on input
        this.autoResizeTextarea();
    }

    handlePaste(event) {
        // Handle paste events
        setTimeout(() => {
            this.updateCharCount();
            this.updateSendButton();
            this.autoResizeTextarea();
        }, 10);
    }

    /**
     * Auto-resize textarea
     */
    autoResizeTextarea() {
        const textarea = this.elements.messageInput;
        if (!textarea) return;
        
        textarea.style.height = 'auto';
        const newHeight = Math.min(textarea.scrollHeight, 120); // Max 120px
        textarea.style.height = newHeight + 'px';
    }

    /**
     * Update character count
     */
    updateCharCount() {
        const count = this.elements.messageInput?.value.length || 0;
        if (this.elements.charCount) {
            this.elements.charCount.textContent = count;
            
            // Color coding
            if (count > this.config.maxMessageLength * 0.9) {
                this.elements.charCount.style.color = '#ef4444';
            } else if (count > this.config.maxMessageLength * 0.8) {
                this.elements.charCount.style.color = '#f59e0b';
            } else {
                this.elements.charCount.style.color = '';
            }
        }
    }

    /**
     * Update send button state
     */
    updateSendButton() {
        const hasText = this.elements.messageInput?.value.trim().length > 0;
        const isConnected = this.state.isConnected;
        
        if (this.elements.sendBtn) {
            this.elements.sendBtn.disabled = !hasText || !isConnected;
            this.elements.sendBtn.classList.toggle('active', hasText && isConnected);
        }
    }

    /**
     * Sidebar management
     */
    toggleSidebar() {
        const sidebar = this.elements.sidebar;
        if (!sidebar) return;
        
        const isCollapsed = sidebar.classList.contains('collapsed');
        
        if (isCollapsed) {
            this.openSidebar();
        } else {
            this.closeSidebar();
        }
    }

    openSidebar() {
        this.elements.sidebar?.classList.remove('collapsed');
        this.elements.mobileOverlay?.classList.add('active');
        document.body.style.overflow = this.state.isMobile ? 'hidden' : '';
    }

    closeSidebar() {
        this.elements.sidebar?.classList.add('collapsed');
        this.elements.mobileOverlay?.classList.remove('active');
        document.body.style.overflow = '';
    }

    /**
     * Theme management
     */
    toggleTheme() {
        this.state.theme = this.state.theme === 'dark' ? 'light' : 'dark';
        this.applyTheme(this.state.theme);
        this.saveUserPreferences();
    }

    applyTheme(theme) {
        document.documentElement.setAttribute('data-theme', theme);
        
        // Update theme toggle icon
        const themeToggle = this.elements.themeToggle;
        if (themeToggle) {
            themeToggle.classList.toggle('light', theme === 'light');
        }
    }

    /**
     * Connection status updates
     */
    updateConnectionStatus(isConnected) {
        const statusElement = this.elements.connectionStatus;
        const textElement = this.elements.connectionText;
        
        if (statusElement) {
            statusElement.className = `status-indicator ${isConnected ? 'connected' : 'disconnected'}`;
        }
        
        if (textElement) {
            textElement.textContent = isConnected ? 'Connected' : 'Disconnected';
        }
    }

    /**
     * Update application statistics
     */
    updateStats() {
        if (this.elements.messageCount) {
            this.elements.messageCount.textContent = this.state.messageCount;
        }
        
        if (this.elements.searchCount) {
            this.elements.searchCount.textContent = this.state.searchCount;
        }
    }

    /**
     * Suggestion chip handling
     */
    handleSuggestionClick(event) {
        const suggestion = event.currentTarget.dataset.suggestion;
        if (suggestion && this.elements.messageInput) {
            this.elements.messageInput.value = suggestion;
            this.updateCharCount();
            this.updateSendButton();
            this.focusInput();
            this.sendMessage();
        }
    }

    /**
     * Utility functions
     */
    hideWelcomeScreen() {
        const welcomeScreen = this.elements.welcomeScreen;
        if (welcomeScreen && !welcomeScreen.classList.contains('hidden')) {
            welcomeScreen.style.opacity = '0';
            welcomeScreen.style.transform = 'translateY(-20px)';
            
            setTimeout(() => {
                welcomeScreen.classList.add('hidden');
            }, this.config.animationDuration);
        }
    }

    hideLoadingScreen() {
        const loadingScreen = this.elements.loadingScreen;
        if (loadingScreen) {
            loadingScreen.style.opacity = '0';
            setTimeout(() => {
                loadingScreen.style.display = 'none';
            }, 500);
        }
    }

    scrollToBottom() {
        const chatMessages = this.elements.chatMessages;
        if (chatMessages) {
            chatMessages.scrollTop = chatMessages.scrollHeight;
        }
    }

    focusInput() {
        this.elements.messageInput?.focus();
    }

    formatTimestamp(date) {
        return new Intl.DateTimeFormat('en-US', {
            hour: '2-digit',
            minute: '2-digit',
            hour12: true
        }).format(date);
    }

    showError(message) {
        console.error('Error:', message);
        
        // Create error toast (simple implementation)
        const toast = document.createElement('div');
        toast.className = 'error-toast';
        toast.textContent = message;
        toast.style.cssText = `
            position: fixed;
            top: 20px;
            right: 20px;
            background: #ef4444;
            color: white;
            padding: 12px 20px;
            border-radius: 8px;
            box-shadow: 0 4px 12px rgba(0,0,0,0.15);
            z-index: 10000;
            opacity: 0;
            transform: translateX(100%);
            transition: all 0.3s ease;
        `;
        
        document.body.appendChild(toast);
        
        // Animate in
        setTimeout(() => {
            toast.style.opacity = '1';
            toast.style.transform = 'translateX(0)';
        }, 10);
        
        // Auto remove
        setTimeout(() => {
            toast.style.opacity = '0';
            toast.style.transform = 'translateX(100%)';
            setTimeout(() => toast.remove(), 300);
        }, 5000);
    }

    /**
     * Event handlers for additional features
     */
    handleResize() {
        this.state.isMobile = window.innerWidth <= 768;
        
        // Close sidebar on mobile when resizing to desktop
        if (!this.state.isMobile) {
            this.elements.mobileOverlay?.classList.remove('active');
            document.body.style.overflow = '';
        }
    }

    handleBeforeUnload(event) {
        if (this.state.websocket && this.state.websocket.readyState === WebSocket.OPEN) {
            this.state.websocket.close(1000, 'Page unload');
        }
    }

    handleOnline() {
        console.log('Connection restored');
        if (!this.state.isConnected) {
            this.connectWebSocket();
        }
    }

    handleOffline() {
        console.log('Connection lost');
        this.updateConnectionStatus(false);
    }

    /**
     * Session management
     */
    createNewSession() {
        // Clear current chat
        const messages = this.elements.chatMessages?.querySelectorAll('.message');
        messages?.forEach(msg => msg.remove());
        
        // Show welcome screen
        this.elements.welcomeScreen?.classList.remove('hidden');
        if (this.elements.welcomeScreen) {
            this.elements.welcomeScreen.style.opacity = '1';
            this.elements.welcomeScreen.style.transform = 'translateY(0)';
        }
        
        // Generate new session
        this.generateSessionId();
        this.connectWebSocket();
        
        // Reset stats
        this.state.messageCount = 0;
        this.updateStats();
        
        this.closeSidebar();
    }

    exportConversation() {
        // Simple export functionality
        const messages = Array.from(this.elements.chatMessages?.querySelectorAll('.message') || []);
        const conversation = messages.map(msg => {
            const isUser = msg.classList.contains('user-message');
            const text = msg.querySelector('.message-text')?.textContent || '';
            const timestamp = msg.querySelector('.message-timestamp')?.textContent || '';
            return `${isUser ? 'User' : 'AI'} (${timestamp}): ${text}`;
        }).join('\n\n');
        
        const blob = new Blob([conversation], { type: 'text/plain' });
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `maverick-conversation-${new Date().toISOString().split('T')[0]}.txt`;
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        URL.revokeObjectURL(url);
    }

    clearAllSessions() {
        if (confirm('Are you sure you want to clear all conversations? This action cannot be undone.')) {
            this.createNewSession();
        }
    }

    handleFilterClick(event) {
        // Remove active class from all filter buttons
        this.elements.filterBtns?.forEach(btn => btn.classList.remove('active'));
        
        // Add active class to clicked button
        event.currentTarget.classList.add('active');
        
        // Filter logic would go here in a full implementation
    }

    handleSessionSearch(event) {
        const query = event.target.value.toLowerCase();
        // Search logic would go here in a full implementation
        console.log('Searching sessions for:', query);
    }

    handleVoiceInput() {
        // Voice input would be implemented here
        this.showError('Voice input feature coming soon!');
    }

    handleFileAttachment() {
        // File attachment would be implemented here
        this.showError('File attachment feature coming soon!');
    }

    openSettings() {
        // Settings modal would be implemented here
        this.showError('Settings panel coming soon!');
    }

    /**
     * User preferences
     */
    loadUserPreferences() {
        try {
            const preferences = JSON.parse(localStorage.getItem('maverick-preferences') || '{}');
            this.state.theme = preferences.theme || 'dark';
        } catch (error) {
            console.warn('Failed to load user preferences:', error);
        }
    }

    saveUserPreferences() {
        try {
            const preferences = {
                theme: this.state.theme
            };
            localStorage.setItem('maverick-preferences', JSON.stringify(preferences));
        } catch (error) {
            console.warn('Failed to save user preferences:', error);
        }
    }

    /**
     * Utility function for debouncing
     */
    debounce(func, wait) {
        let timeout;
        return function executedFunction(...args) {
            const later = () => {
                clearTimeout(timeout);
                func(...args);
            };
            clearTimeout(timeout);
            timeout = setTimeout(later, wait);
        };
    }
}

// Initialize application when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    // Initialize Maverick AI
    window.maverickAI = new MaverickAI();
});

// Handle page visibility changes
document.addEventListener('visibilitychange', () => {
    if (document.hidden) {
        console.log('Page hidden');
    } else {
        console.log('Page visible');
        // Reconnect if needed
        if (window.maverickAI && !window.maverickAI.state.isConnected) {
            window.maverickAI.connectWebSocket();
        }
    }
});

// Global error handler
window.addEventListener('error', (event) => {
    console.error('Global error:', event.error);
    if (window.maverickAI) {
        window.maverickAI.showError('An unexpected error occurred');
    }
});

// Handle unhandled promise rejections
window.addEventListener('unhandledrejection', (event) => {
    console.error('Unhandled promise rejection:', event.reason);
    event.preventDefault();
    if (window.maverickAI) {
        window.maverickAI.showError('Connection error occurred');
    }
});

console.log('🤖 Maverick AI Frontend Script Loaded - v2.1.0');