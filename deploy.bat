@echo off
REM ParkVision Docker Deployment Script for Windows

setlocal enabledelayedexpansion

REM Configuration
set IMAGE_NAME=parkvision
set CONTAINER_NAME=parkvision-app
set PORT=5000

REM Colors (if supported)
set "GREEN=[92m"
set "YELLOW=[93m"
set "RED=[91m"
set "NC=[0m"

REM Functions
:print_status
echo %GREEN%[INFO]%NC% %~1
goto :eof

:print_warning
echo %YELLOW%[WARNING]%NC% %~1
goto :eof

:print_error
echo %RED%[ERROR]%NC% %~1
goto :eof

:check_docker
docker --version >nul 2>&1
if %errorlevel% neq 0 (
    call :print_error "Docker is not installed. Please install Docker Desktop first."
    exit /b 1
)
call :print_status "Docker is installed"
goto :eof

:check_docker_compose
docker-compose --version >nul 2>&1
if %errorlevel% neq 0 (
    call :print_warning "Docker Compose is not installed. Using Docker commands instead."
    goto :eof
)
call :print_status "Docker Compose is available"
goto :eof

:build_image
call :print_status "Building ParkVision Docker image..."
docker build -t %IMAGE_NAME%:latest .
if %errorlevel% neq 0 (
    call :print_error "Failed to build Docker image"
    exit /b 1
)
call :print_status "Image built successfully"
goto :eof

:cleanup_container
docker ps -a --format "table {{.Names}}" | findstr %CONTAINER_NAME% >nul 2>&1
if %errorlevel% equ 0 (
    call :print_status "Stopping and removing existing container..."
    docker stop %CONTAINER_NAME% >nul 2>&1
    docker rm %CONTAINER_NAME% >nul 2>&1
)
goto :eof

:run_with_compose
call :print_status "Starting ParkVision with Docker Compose..."
docker-compose down >nul 2>&1
docker-compose up -d
if %errorlevel% neq 0 (
    call :print_error "Failed to start with Docker Compose"
    exit /b 1
)
call :print_status "ParkVision is running at http://localhost:%PORT%"
goto :eof

:run_with_docker
call :print_status "Starting ParkVision container..."
call :cleanup_container

docker run -d ^
    --name %CONTAINER_NAME% ^
    -p %PORT%:5000 ^
    -e PYTHONUNBUFFERED=1 ^
    -e FLASK_ENV=production ^
    -e DISPLAY_ENABLED=false ^
    --restart unless-stopped ^
    %IMAGE_NAME%:latest

if %errorlevel% neq 0 (
    call :print_error "Failed to start Docker container"
    exit /b 1
)
call :print_status "ParkVision is running at http://localhost:%PORT%"
goto :eof

:show_logs
docker-compose --version >nul 2>&1
if %errorlevel% equ 0 (
    docker-compose logs -f
) else (
    docker logs -f %CONTAINER_NAME%
)
goto :eof

:stop_app
docker-compose --version >nul 2>&1
if %errorlevel% equ 0 (
    call :print_status "Stopping ParkVision with Docker Compose..."
    docker-compose down
) else (
    call :print_status "Stopping ParkVision container..."
    docker stop %CONTAINER_NAME% >nul 2>&1
    docker rm %CONTAINER_NAME% >nul 2>&1
)
call :print_status "ParkVision stopped"
goto :eof

REM Main script
set COMMAND=%1
if "%COMMAND%"=="" set COMMAND=deploy

if "%COMMAND%"=="build" (
    call :check_docker
    call :build_image
) else if "%COMMAND%"=="deploy" (
    call :check_docker
    call :build_image
    call :check_docker_compose
    if %errorlevel% equ 0 (
        call :run_with_compose
    ) else (
        call :run_with_docker
    )
) else if "%COMMAND%"=="start" (
    call :check_docker
    call :check_docker_compose
    if %errorlevel% equ 0 (
        call :run_with_compose
    ) else (
        call :run_with_docker
    )
) else if "%COMMAND%"=="stop" (
    call :stop_app
) else if "%COMMAND%"=="logs" (
    call :show_logs
) else if "%COMMAND%"=="restart" (
    call :stop_app
    timeout /t 2 /nobreak >nul
    call :check_docker_compose
    if %errorlevel% equ 0 (
        call :run_with_compose
    ) else (
        call :run_with_docker
    )
) else (
    echo ParkVision Docker Deployment Script for Windows
    echo.
    echo Usage: %0 [command]
    echo.
    echo Commands:
    echo   build    - Build the Docker image
    echo   deploy   - Build and start the application (default)
    echo   start    - Start the application
    echo   stop     - Stop the application
    echo   restart  - Restart the application
    echo   logs     - Show application logs
    echo.
    echo Examples:
    echo   %0 deploy    # Build and start the application
    echo   %0 logs      # View application logs
    echo   %0 stop      # Stop the application
)

endlocal
