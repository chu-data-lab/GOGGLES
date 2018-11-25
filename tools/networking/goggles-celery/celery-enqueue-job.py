import os
import socket
import sys
import time

from celery import Celery

APP_NAME = 'celery-daemon'
TASK_NAME = 'train'
STABLE_STATE = 'TRAINING'
BROKER_URL = os.environ['GOGGLES_BROKER_URL']
BACKEND_URL = os.environ['GOGGLES_BACKEND_URL']


app = Celery(APP_NAME, broker=BROKER_URL, backend=BACKEND_URL)


if __name__ == '__main__':
    arg_str = ' '.join(sys.argv[1:])

    result = app.send_task(
        '%s.%s' % (APP_NAME, TASK_NAME),
        args=(arg_str,))

    start, timeout = time.time(), 60
    while True:
        now = time.time()
        if (now - start > timeout
                or result.state == STABLE_STATE):
            print(result.info)
            break
        time.sleep(1)
