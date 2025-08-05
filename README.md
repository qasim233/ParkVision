# ParkVision Docker Deployment

This repository contains the Docker configuration files for deploying the ParkVision parking management system.

## Quick Start

### Prerequisites

- Docker Desktop (Windows/Mac) or Docker Engine (Linux)
- Docker Compose (usually included with Docker Desktop)

### Deployment Options

#### Option 1: Using Deployment Scripts (Recommended)

**For Windows:**
```powershell
# Navigate to the project directory
cd "c:\Users\Admin\Desktop\deploy_ParkVision"

# Deploy the application
.\deploy.bat deploy
```

**For Linux/Mac:**
```bash
# Navigate to the project directory
cd /path/to/deploy_ParkVision

# Make the script executable
chmod +x deploy.sh

# Deploy the application
./deploy.sh deploy
```

#### Option 2: Using Docker Compose

```bash
# Build and start the application
docker-compose up -d

# View logs
docker-compose logs -f

# Stop the application
docker-compose down
```

#### Option 3: Using Docker Commands

```bash
# Build the image
docker build -t parkvision:latest .

# Run the container
docker run -d \
  --name parkvision-app \
  -p 5000:5000 \
  -e PYTHONUNBUFFERED=1 \
  -e FLASK_ENV=production \
  --restart unless-stopped \
  parkvision:latest
```

## Application Access

Once deployed, the application will be available at:
- **Main Application**: http://localhost:5000
- **Health Check**: http://localhost:5000/health
- **API Endpoints**: http://localhost:5000/snapshot, http://localhost:5000/stream

## Available Commands

### Deployment Script Commands

| Command | Description |
|---------|-------------|
| `deploy` | Build and start the application (default) |
| `build` | Build the Docker image only |
| `start` | Start the application |
| `stop` | Stop the application |
| `restart` | Restart the application |
| `logs` | Show application logs |

### Examples

```bash
# Windows
.\deploy.bat build     # Build image only
.\deploy.bat start     # Start application
.\deploy.bat logs      # View logs
.\deploy.bat stop      # Stop application
.\deploy.bat restart   # Restart application

# Linux/Mac
./deploy.sh build      # Build image only
./deploy.sh start      # Start application
./deploy.sh logs       # View logs
./deploy.sh stop       # Stop application
./deploy.sh restart    # Restart application
```

## Project Structure

```
deploy_ParkVision/
├── app.py                           # Main Flask application
├── parking_detector.py              # Parking detection logic
├── pathfinding.py                   # Pathfinding algorithms
├── requirements.txt                 # Python dependencies
├── yolo12n.pt                      # YOLO model weights
├── parking_slots_Collage_Image.json # Parking slot configuration
├── image4.png                      # Sample image
├── images/                         # Image assets
├── Dockerfile                      # Docker image configuration
├── docker-compose.yml              # Docker Compose configuration
├── .dockerignore                   # Docker build ignore rules
├── deploy.sh                       # Linux/Mac deployment script
├── deploy.bat                      # Windows deployment script
└── README.md                       # This file
```

## Configuration

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `PYTHONUNBUFFERED` | `1` | Disable Python output buffering |
| `FLASK_ENV` | `production` | Flask environment |
| `DISPLAY_ENABLED` | `false` | Enable/disable OpenCV display windows |

### Port Configuration

The application runs on port 5000 by default. You can change this by modifying:
- `docker-compose.yml`: Change the ports mapping
- Deployment scripts: Update the `PORT` variable

### Volume Mounts

The Docker setup includes volume mounts for:
- `./logs:/app/logs` - Application logs
- `./temp:/app/temp` - Temporary files

## Troubleshooting

### Common Issues

1. **Port already in use**
   ```bash
   # Check what's using port 5000
   netstat -ano | findstr :5000    # Windows
   lsof -i :5000                   # Linux/Mac
   
   # Stop the conflicting process or change the port
   ```

2. **Docker not found**
   - Install Docker Desktop from https://www.docker.com/products/docker-desktop

3. **Permission denied (Linux/Mac)**
   ```bash
   # Make sure the script is executable
   chmod +x deploy.sh
   
   # Or run with bash
   bash deploy.sh deploy
   ```

4. **Build failures**
   ```bash
   # Clean Docker cache and rebuild
   docker system prune -a
   docker build --no-cache -t parkvision:latest .
   ```

### Viewing Logs

```bash
# Using deployment script
./deploy.sh logs        # Linux/Mac
.\deploy.bat logs       # Windows

# Using Docker Compose
docker-compose logs -f

# Using Docker commands
docker logs -f parkvision-app
```

### Health Check

The application includes a health check endpoint:
```bash
curl http://localhost:5000/health
```

Expected response:
```json
{
  "status": "healthy",
  "timestamp": "2025-01-XX",
  "uptime": "XX seconds"
}
```

## Development

### Building for Development

```bash
# Build with development settings
docker build -t parkvision:dev --build-arg ENV=development .

# Run with volume mounting for live code changes
docker run -d \
  --name parkvision-dev \
  -p 5000:5000 \
  -v $(pwd):/app \
  -e FLASK_ENV=development \
  parkvision:dev
```

### Debugging

```bash
# Run interactively
docker run -it --rm parkvision:latest /bin/bash

# Check container status
docker ps
docker inspect parkvision-app
```

## Production Considerations

1. **Security**: Consider using HTTPS with a reverse proxy (nginx, traefik)
2. **Monitoring**: Add monitoring solutions (Prometheus, Grafana)
3. **Backup**: Backup your parking slot configuration and model files
4. **Scaling**: Use Docker Swarm or Kubernetes for multiple instances
5. **Updates**: Implement a CI/CD pipeline for automated deployments

## Support

If you encounter issues:
1. Check the logs using the commands above
2. Verify Docker is running properly
3. Ensure all required files are present
4. Check system resources (CPU, memory, disk space)

For more detailed information about the ParkVision system, refer to the main project documentation.
