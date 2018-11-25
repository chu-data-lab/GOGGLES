# Celery Daemon for GOGGLES

This module implements a Celery daemon
that allows models to be trained parallely
and in a distributed manner. The daemon connects
to a globally hosted rabbitmq broker and a redis backend
to listen for jobs. When the daemon recieves a job,
it starts the training inside of a tmux session. It keeps
track of whether the job is still running inside of
the session through a lock file placed in
the `GOGGLES/_scratch/locks/` directory. When execution
of the training script finishes, the lock file is released,
and the celery task is marked as finished. This allows having
multiple workers with separate training jobs to run
concurrently. The concurrency of a daemon running
on a server can be configured depending on the resources
available for that server (e.g., GPU memory).

## USAGE

### Set up the environment
```bash
export GOGGLES_ENV="goggles"
export GOGGLES_BASE_DIR="/path/to/GOGGLES"
export GOGGLES_NETWORK_HOST="127.0.0.1"
export GOGGLES_BROKER_URL="amqp://admin:password@${GOGGLES_NETWORK_HOST}:5672/goggles"
export GOGGLES_BACKEND_URL="redis://${GOGGLES_NETWORK_HOST}:6379"

source activate ${GOGGLES_ENV}
```

### Start the celery daemon
```bash
celery worker \
    -A celery-daemon \
    -l info \
    -Ofair \
    -c 2
```

### Queue jobs
```bash
python celery-enqueue-job.py with \
    num_epochs=12000 \
    filter_class_ids=14,90
```
