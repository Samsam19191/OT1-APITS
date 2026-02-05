.PHONY: help setup-backend setup-db-app setup-db-eval setup-frontend run-app eval rebuild clean

# Default target
help:
	@echo "Available targets:"
	@echo "  make run-app        - Setup and run the application (DB + Backend + Frontend)"
	@echo "  make eval           - Setup and run evaluation pipeline"
	@echo "  make rebuild        - Clean and rebuild everything"
	@echo "  make clean          - Stop containers and clean up"
	@echo ""
	@echo "Low-level targets:"
	@echo "  make setup-backend  - Setup Python environment"
	@echo "  make setup-db-app   - Setup and seed application database"
	@echo "  make setup-db-eval  - Setup and seed evaluation database"
	@echo "  make setup-frontend - Setup frontend dependencies"

# === Low-level targets ===

setup-backend:
	@echo "Setting up backend environment..."
	@python3 --version | grep -q "Python 3.1[0-9]" || (echo "Python 3.10+ required" && exit 1)
	@if [ ! -d ".venv" ]; then \
		echo "Creating virtual environment..."; \
		python3 -m venv .venv; \
	fi
	@echo "Installing dependencies..."
	@.venv/bin/pip install -q --upgrade pip
	@.venv/bin/pip install -q -r requirements.txt
	@echo "Backend environment ready"

setup-db-app: setup-backend
	@echo "Setting up application database..."
	@docker-compose -f docker-compose.app.yml up -d db_app
	@sleep 2  # Wait for DB to be ready
	@.venv/bin/python scripts/seed_app.py
	@echo "Application database ready"

setup-db-eval: setup-backend
	@echo "Setting up evaluation database..."
	@docker-compose -f docker-compose.eval.yml up -d db_eval
	@sleep 2  # Wait for DB to be ready
	@.venv/bin/python eval/seed_eval_db.py
	@echo "Evaluation database ready"

setup-frontend:
	@echo "Setting up frontend..."
	@cd ui && npm install
	@echo "Frontend ready"

# === High-level targets ===

run-app: setup-backend setup-db-app setup-frontend
	@echo "Starting application..."
	@docker-compose -f docker-compose.app.yml up -d
	@echo ""
	@echo "Starting backend server (uvicorn) and frontend (vite)..."
	@echo "Press Ctrl+C to stop all processes"
	@echo ""
	@trap 'docker-compose -f docker-compose.app.yml stop; exit 0' INT; \
	(.venv/bin/python run_server.py & \
	 cd ui && npm run dev) || docker-compose -f docker-compose.app.yml stop

eval: setup-backend setup-db-eval
	@echo "Running evaluation pipeline..."
	@.venv/bin/python eval/eval.py
	@echo "Evaluation complete. Check eval/data/reports/ for results"

rebuild: clean
	@echo "Rebuilding everything..."
	@$(MAKE) setup-backend
	@$(MAKE) setup-frontend
	@docker-compose -f docker-compose.app.yml build --no-cache
	@echo "Rebuild complete"

clean:
	@echo "Cleaning up..."
	@docker-compose -f docker-compose.app.yml down -v 2>/dev/null || true
	@docker-compose -f docker-compose.eval.yml down -v 2>/dev/null || true
	@rm -rf .venv
	@rm -rf ui/node_modules
	@echo "Cleanup complete"
