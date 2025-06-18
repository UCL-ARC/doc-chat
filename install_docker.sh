#!/bin/bash

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to print status messages
print_status() {
    echo -e "${YELLOW}[*] $1${NC}"
}

# Function to print success messages
print_success() {
    echo -e "${GREEN}[+] $1${NC}"
}

# Function to print error messages
print_error() {
    echo -e "${RED}[-] $1${NC}"
}

# Function to check if command was successful
check_status() {
    if [ $? -eq 0 ]; then
        print_success "$1"
    else
        print_error "$2"
        exit 1
    fi
}

# Check if script is run as root
if [ "$EUID" -ne 0 ]; then
    print_error "Please run this script as root or with sudo"
    exit 1
fi

# Detect OS
if [ -f /etc/os-release ]; then
    . /etc/os-release
    OS=$ID
else
    print_error "Cannot detect OS"
    exit 1
fi

print_status "Installing Docker and Docker Compose on $OS..."

# Install Docker based on OS
case $OS in
    "ubuntu"|"debian")
        # Update package index
        print_status "Updating package index..."
        apt-get update
        check_status "Package index updated" "Failed to update package index"

        # Install prerequisites
        print_status "Installing prerequisites..."
        apt-get install -y \
            apt-transport-https \
            ca-certificates \
            curl \
            gnupg \
            lsb-release
        check_status "Prerequisites installed" "Failed to install prerequisites"

        # Add Docker's official GPG key
        print_status "Adding Docker's GPG key..."
        curl -fsSL https://download.docker.com/linux/$OS/gpg | gpg --dearmor -o /usr/share/keyrings/docker-archive-keyring.gpg
        check_status "Docker GPG key added" "Failed to add Docker GPG key"

        # Set up the stable repository
        print_status "Setting up Docker repository..."
        echo \
        "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/docker-archive-keyring.gpg] https://download.docker.com/linux/$OS \
        $(lsb_release -cs) stable" | tee /etc/apt/sources.list.d/docker.list > /dev/null
        check_status "Docker repository added" "Failed to add Docker repository"

        # Install Docker Engine
        print_status "Installing Docker Engine..."
        apt-get update
        apt-get install -y docker-ce docker-ce-cli containerd.io
        check_status "Docker Engine installed" "Failed to install Docker Engine"
        ;;

    "fedora")
        # Install prerequisites
        print_status "Installing prerequisites..."
        dnf -y install dnf-plugins-core
        check_status "Prerequisites installed" "Failed to install prerequisites"

        # Add Docker repository
        print_status "Adding Docker repository..."
        dnf config-manager --add-repo https://download.docker.com/linux/fedora/docker-ce.repo
        check_status "Docker repository added" "Failed to add Docker repository"

        # Install Docker Engine
        print_status "Installing Docker Engine..."
        dnf -y install docker-ce docker-ce-cli containerd.io
        check_status "Docker Engine installed" "Failed to install Docker Engine"
        ;;

    *)
        print_error "Unsupported OS: $OS"
        exit 1
        ;;
esac

# Install Docker Compose
print_status "Installing Docker Compose..."
COMPOSE_VERSION=$(curl -s https://api.github.com/repos/docker/compose/releases/latest | grep 'tag_name' | cut -d'"' -f4)
curl -L "https://github.com/docker/compose/releases/download/${COMPOSE_VERSION}/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
check_status "Docker Compose downloaded" "Failed to download Docker Compose"

chmod +x /usr/local/bin/docker-compose
check_status "Docker Compose permissions set" "Failed to set Docker Compose permissions"

# Start and enable Docker service
print_status "Starting Docker service..."
systemctl start docker
systemctl enable docker
check_status "Docker service started and enabled" "Failed to start Docker service"

# Add current user to docker group
if [ ! -z "$SUDO_USER" ]; then
    print_status "Adding user $SUDO_USER to docker group..."
    usermod -aG docker $SUDO_USER
    check_status "User added to docker group" "Failed to add user to docker group"
fi

# Verify installations
print_status "Verifying installations..."
DOCKER_VERSION=$(docker --version)
COMPOSE_VERSION=$(docker-compose --version)

print_success "Docker installed successfully: $DOCKER_VERSION"
print_success "Docker Compose installed successfully: $COMPOSE_VERSION"

print_status "Installation complete!"
echo -e "${YELLOW}NOTE: You may need to log out and back in for group changes to take effect.${NC}"
