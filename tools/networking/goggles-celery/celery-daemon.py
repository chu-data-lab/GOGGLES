import os
import socket
import sys
import time

from celery import Celery
import libtmux
from libtmux.exc import TmuxSessionExists


APP_NAME = 'celery-daemon'
BROKER_URL = os.environ['GOGGLES_BROKER_URL']
BACKEND_URL = os.environ['GOGGLES_BACKEND_URL']
GOGGLES_ENV = os.environ['GOGGLES_ENV']
GOGGLES_BASE_DIR = os.environ['GOGGLES_BASE_DIR']
assert os.path.isdir(GOGGLES_BASE_DIR)

app = Celery(APP_NAME, broker=BROKER_URL, backend=BACKEND_URL)
new_session_id = lambda: 'goggles-%s' % (time.strftime('%Y%m%dT%H%M%S'))


@app.task(bind=True)
def train(self, train_args):
    print('Starting training %s' % train_args)

    LOCK_FILE = None
    LOCK_DIR = os.path.join(
        GOGGLES_BASE_DIR, '_scratch', 'locks')
    if not os.path.exists(LOCK_DIR):
        os.makedirs(LOCK_DIR)

    TRAINING_SRC_PATH = os.path.join(
        GOGGLES_BASE_DIR, 'goggles', 'train.py')
    assert os.path.isfile(TRAINING_SRC_PATH)

    ACTIVATE_ENV_CMD = 'source activate %s' % GOGGLES_ENV
    RUN_TRAINING_CMD = 'python3 {training_src_path} {train_args}'.format(
        training_src_path=TRAINING_SRC_PATH,
        train_args=train_args)

    server = libtmux.Server()
    hostname = socket.gethostname()
    session, session_id = None, None
    while True:
        session_id = new_session_id()
        try:
            session = server.new_session(session_id)

            LOCK_FILE = os.path.join(LOCK_DIR, '%s.lock' % session_id)
            with open(LOCK_FILE, 'w') as f:
                f.write(self.request.id)

        except TmuxSessionExists:
            time.sleep(2)

        else:
            break

    assert session is not None
    assert os.path.isfile(LOCK_FILE)

    session.attached_pane.send_keys(ACTIVATE_ENV_CMD)
    time.sleep(4)  # Waiting for env to get activated
    session.attached_pane.send_keys('%s; %s' % (
        RUN_TRAINING_CMD,
        'rm -f %s' % LOCK_FILE))

    while os.path.exists(LOCK_FILE):
        self.update_state(
            state='TRAINING',
            meta={'hostname': hostname,
                  'session_id': session_id})

        time.sleep(20)

    return hostname, session_id
