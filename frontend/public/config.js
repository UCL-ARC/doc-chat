// Use current origin so API works on any port (e.g. 8001, 8002)
window.APP_CONFIG = {
    API_URL: window.location.hostname === 'localhost' || window.location.hostname === '0.0.0.0'
        ? window.location.origin
        : 'https://workshop-gpt-b2gqg0c8g7b5bwa4.eastus-01.azurewebsites.net'
}; 