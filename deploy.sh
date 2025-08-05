#!/bin/bash

# ParkVision Docker Deployment Script

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Configuration
IMAGE_NAME="parkvision"
CONTAINER_NAME="parkvision-app"
PORT="5000"

# Functions
print_status() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if Docker is installed
check_docker() {
    if ! command -v docker &> /dev/null; then
        print_error "Docker is not installed. Please install Docker first."
        exit 1
    fi
    print_status "Docker is installed"
}

# Check if Docker Compose is installed
check_docker_compose() {
    if ! command -v docker-compose &> /dev/null; then
        print_warning "Docker Compose is not installed. Using Docker commands instead."
        return 1
    fi
    print_status "Docker Compose is available"
    return 0
}

# Build the Docker image
build_image() {
    print_status "Building ParkVision Docker image..."
    docker build -t $IMAGE_NAME:latest .
    print_status "Image built successfully"
}

# Stop and remove existing container
cleanup_container() {
    if docker ps -a --format 'table {{.Names}}' | grep -q $CONTAINER_NAME; then
        print_status "Stopping and removing existing container..."
        docker stop $CONTAINER_NAME || true
        docker rm $CONTAINER_NAME || true
    fi
}

# Run with Docker Compose
run_with_compose() {
    print_status "Starting ParkVision with Docker Compose..."
    docker-compose down || true
    docker-compose up -d
    print_status "ParkVision is running at http://localhost:$PORT"
}

# Run with Docker
run_with_docker() {
    print_status "Starting ParkVision container..."
    cleanup_container
    
    docker run -d \
        --name $CONTAINER_NAME \
        -p $PORT:5000 \
        -e PYTHONUNBUFFERED=1 \
        -e FLASK_ENV=production \
        -e DISPLAY_ENABLED=false \
        --restart unless-stopped \
        $IMAGE_NAME:latest
    
    print_status "ParkVision is running at http://localhost:$PORT"
}

# Show logs
show_logs() {
    if check_docker_compose; then
        docker-compose logs -f
    else
        docker logs -f $CONTAINER_NAME
    fi
}

# Stop the application
stop_app() {
    if check_docker_compose; then
        print_status "Stopping ParkVision with Docker Compose..."
        docker-compose down
    else
        print_status "Stopping ParkVision container..."
        docker stop $CONTAINER_NAME || true
        docker rm $CONTAINER_NAME || true
    fi
    print_status "ParkVision stopped"
}

# Main script
main() {
    case "${1:-deploy}" in
        "build")
            check_docker
            build_image
            ;;
        "deploy")
            check_docker
            build_image
            if check_docker_compose; then
                run_with_compose
            else
                run_with_docker
            fi
            ;;
        "start")
            check_docker
            if check_docker_compose; then
                run_with_compose
            else
                run_with_docker
            fi
            ;;
        "stop")
            stop_app
            ;;
        "logs")
            show_logs
            ;;
        "restart")
            stop_app
            sleep 2
            if check_docker_compose; then
                run_with_compose
            else
                run_with_docker
            fi
            ;;
        *)
            echo "ParkVision Docker Deployment Script"
            echo ""
            echo "Usage: $0 [command]"
            echo ""
            echo "Commands:"
            echo "  build    - Build the Docker image"
            echo "  deploy   - Build and start the application (default)"
            echo "  start    - Start the application"
            echo "  stop     - Stop the application"
            echo "  restart  - Restart the application"
            echo "  logs     - Show application logs"
            echo ""
            echo "Examples:"
            echo "  $0 deploy    # Build and start the application"
            echo "  $0 logs      # View application logs"
            echo "  $0 stop      # Stop the application"
            ;;
    esac
}

# Run the main function
main "$@"
