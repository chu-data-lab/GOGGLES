### Setting up the environment
```bash
export GOGGLES_ENV="goggles"
export GOGGLES_BASE_DIR="/path/to/GOGGLES"
export GOGGLES_NETWORK_HOST="127.0.0.1"
export GOGGLES_BROKER_URL="amqp://admin:password@${GOGGLES_NETWORK_HOST}:5672/goggles"
export GOGGLES_BACKEND_URL="redis://${GOGGLES_NETWORK_HOST}:6379"
```

### Starting the celery daemon
```bash
celery worker \
  -A goggles-celery \
  -l info \
  -Ofair \
  -c 2
```

### Queueing jobs
```bash
python goggles-celery.py $id1 $id2 ...
```
