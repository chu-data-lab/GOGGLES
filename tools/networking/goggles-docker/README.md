# Docker Services for GOGGLES

This module provisions docker services
for running rabbitmq and redis for global access.
These services are used by GOGGLES for
brokering the distributed execution of
training jobs across servers.

## USAGE

### Build the services
```bash
docker-compose build
```

### Start the services
```bash
docker-compose up
```
