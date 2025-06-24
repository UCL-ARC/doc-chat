window.APP_CONFIG = {
    API_URL: window.location.hostname === 'localhost' || window.location.hostname === '0.0.0.0'
        ? 'http://localhost:8001'
        : 'https://workshop-gpt-b2gqg0c8g7b5bwa4.eastus-01.azurewebsites.net'
}; 