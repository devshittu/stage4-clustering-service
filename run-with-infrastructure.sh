#!/bin/bash

# =============================================================================
# Stage 4 Clustering Service - Infrastructure Integration Script
# =============================================================================
# This script manages Stage 4 deployment with centralized infrastructure
# Usage: ./run-with-infrastructure.sh [start|stop|restart|logs|status]
# =============================================================================

set -e

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
PROJECT_NAME="stage4-clustering"
COMPOSE_FILE="docker-compose.infrastructure.yml"
INFRASTRUCTURE_DIR="../infrastructure"
STAGE3_DIR="../stage3_embedding_service"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# -----------------------------------------------------------------------------
# Helper Functions
# -----------------------------------------------------------------------------

print_header() {
    echo -e "${BLUE}=====================================================================${NC}"
    echo -e "${BLUE}  Stage 4 Clustering Service - Infrastructure Integration${NC}"
    echo -e "${BLUE}=====================================================================${NC}"
}

print_success() {
    echo -e "${GREEN}✓ $1${NC}"
}

print_error() {
    echo -e "${RED}✗ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠ $1${NC}"
}

print_info() {
    echo -e "${BLUE}ℹ $1${NC}"
}

# Check if infrastructure is running
check_infrastructure() {
    print_info "Checking infrastructure status..."

    if [ ! -d "$INFRASTRUCTURE_DIR" ]; then
        print_error "Infrastructure directory not found: $INFRASTRUCTURE_DIR"
        exit 1
    fi

    cd "$INFRASTRUCTURE_DIR"

    # Check critical services
    local services=("redis-broker" "redis-cache" "postgres" "traefik")
    local all_running=true

    for service in "${services[@]}"; do
        if docker compose ps | grep -q "$service.*running"; then
            print_success "$service is running"
        else
            print_error "$service is not running"
            all_running=false
        fi
    done

    cd - > /dev/null

    if [ "$all_running" = false ]; then
        print_error "Infrastructure is not fully running"
        print_info "Start infrastructure with: cd $INFRASTRUCTURE_DIR && ./scripts/start.sh"
        exit 1
    fi

    print_success "Infrastructure is running"
}

# Check if Stage 3 is running
check_stage3() {
    print_info "Checking Stage 3 (upstream dependency)..."

    if [ ! -d "$STAGE3_DIR" ]; then
        print_warning "Stage 3 directory not found: $STAGE3_DIR"
        print_warning "Stage 4 will not be able to load vector indices"
        return
    fi

    cd "$STAGE3_DIR"

    if docker compose ps | grep -q "embeddings-orchestrator.*running"; then
        print_success "Stage 3 is running"

        # Check if indices exist
        if [ -d "data/vector_indices" ] && [ "$(ls -A data/vector_indices/*.index 2>/dev/null)" ]; then
            print_success "Stage 3 vector indices found"
        else
            print_warning "Stage 3 indices not found - Stage 4 may fail"
        fi
    else
        print_warning "Stage 3 is not running"
        print_info "Start Stage 3 with: cd $STAGE3_DIR && ./run-with-infrastructure.sh start"
    fi

    cd - > /dev/null
}

# Check environment file
check_env() {
    print_info "Checking environment configuration..."

    if [ ! -f .env ]; then
        print_warning ".env file not found"
        print_info "Creating .env from .env.example..."

        if [ -f .env.example ]; then
            cp .env.example .env
            print_success "Created .env file"
            print_warning "Please edit .env and set STAGE4_POSTGRES_PASSWORD"
        else
            print_error ".env.example not found"
            exit 1
        fi
    else
        print_success ".env file found"

        # Check critical variables
        if ! grep -q "^STAGE4_POSTGRES_PASSWORD=" .env || grep -q "^STAGE4_POSTGRES_PASSWORD=your_secure_password_here" .env; then
            print_warning "STAGE4_POSTGRES_PASSWORD not set in .env"
            print_info "Please set a secure password before starting"
        fi
    fi
}

# Create necessary directories
create_directories() {
    print_info "Creating data directories..."

    mkdir -p data/{indices,output,clusters}
    mkdir -p logs
    mkdir -p config

    touch data/indices/.gitkeep
    touch data/output/.gitkeep
    touch data/clusters/.gitkeep

    print_success "Directories created"
}

# Start services
start_services() {
    print_header
    print_info "Starting Stage 4 Clustering Service..."

    check_infrastructure
    check_stage3
    check_env
    create_directories

    print_info "Building and starting containers..."
    docker compose -f "$COMPOSE_FILE" up -d --build

    print_info "Waiting for services to be healthy..."
    sleep 10

    # Check service health
    if docker compose -f "$COMPOSE_FILE" ps | grep -q "clustering-orchestrator.*running"; then
        print_success "Orchestrator started"
    else
        print_error "Orchestrator failed to start"
        docker compose -f "$COMPOSE_FILE" logs orchestrator-service
        exit 1
    fi

    if docker compose -f "$COMPOSE_FILE" ps | grep -q "clustering-celery-worker.*running"; then
        print_success "Celery worker started"
    else
        print_error "Celery worker failed to start"
        docker compose -f "$COMPOSE_FILE" logs celery-worker
        exit 1
    fi

    print_success "Stage 4 Clustering Service started successfully!"
    print_info ""
    print_info "Access via Traefik: http://localhost/api/v1/clustering/health"
    print_info "View logs: ./run-with-infrastructure.sh logs"
    print_info "Check status: ./run-with-infrastructure.sh status"
}

# Stop services
stop_services() {
    print_header
    print_info "Stopping Stage 4 Clustering Service..."

    docker compose -f "$COMPOSE_FILE" down

    print_success "Stage 4 stopped"
}

# Restart services
restart_services() {
    print_header
    print_info "Restarting Stage 4 Clustering Service..."

    stop_services
    sleep 2
    start_services
}

# Show logs
show_logs() {
    print_header
    print_info "Showing Stage 4 logs (Ctrl+C to exit)..."

    docker compose -f "$COMPOSE_FILE" logs -f
}

# Show status
show_status() {
    print_header
    print_info "Stage 4 Clustering Service Status:"
    echo ""

    docker compose -f "$COMPOSE_FILE" ps

    echo ""
    print_info "Health Check:"

    # Check health endpoint
    if curl -sf http://localhost/api/v1/clustering/health > /dev/null 2>&1; then
        print_success "API is healthy"
        curl -s http://localhost/api/v1/clustering/health | jq '.' 2>/dev/null || curl -s http://localhost/api/v1/clustering/health
    else
        print_warning "API health check failed"
        print_info "Service may still be starting up..."
    fi
}

# -----------------------------------------------------------------------------
# Main Script
# -----------------------------------------------------------------------------

case "${1:-}" in
    start)
        start_services
        ;;
    stop)
        stop_services
        ;;
    restart)
        restart_services
        ;;
    logs)
        show_logs
        ;;
    status)
        show_status
        ;;
    *)
        print_header
        echo "Usage: $0 {start|stop|restart|logs|status}"
        echo ""
        echo "Commands:"
        echo "  start    - Start Stage 4 with infrastructure"
        echo "  stop     - Stop Stage 4 services"
        echo "  restart  - Restart Stage 4 services"
        echo "  logs     - Show service logs (follow mode)"
        echo "  status   - Show service status and health"
        echo ""
        exit 1
        ;;
esac
