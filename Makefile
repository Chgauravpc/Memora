.PHONY: help install start stop restart logs demo clean test

help:
	@echo "Long-Form Memory System - Phase 1"
	@echo ""
	@echo "Available commands:"
	@echo "  make install    - Install Python dependencies"
	@echo "  make start      - Start Redis in background"
	@echo "  make stop       - Stop Redis"
	@echo "  make restart    - Restart Redis"
	@echo "  make logs       - Show Redis logs"
	@echo "  make demo       - Run the demo script"
	@echo "  make clean      - Stop services and clean data"
	@echo "  make test       - Run tests (future)"

install:
	pip install -r requirements.txt

start:
	docker-compose up -d
	@echo "✓ Redis started"
	@docker-compose ps

stop:
	docker-compose down
	@echo "✓ Redis stopped"

restart:
	docker-compose restart
	@echo "✓ Redis restarted"
	@docker-compose ps

logs:
	docker-compose logs -f redis

demo:
	python demo.py

clean:
	docker-compose down -v
	rm -rf memory/demo_user/
	@echo "✓ Cleaned up Redis data and demo user memories"

test:
	@echo "Tests not yet implemented (Phase 5)"
	@echo "Run 'make demo' to validate functionality"
