import os
import socket
import sys
import time

from celery import Celery
import libtmux
from libtmux.exc import TmuxSessionExists

BROKER_URL = os.environ['GOGGLES_BROKER_URL']
BACKEND_URL = os.environ['GOGGLES_BACKEND_URL']
GOGGLES_ENV = os.environ['GOGGLES_ENV']
GOGGLES_BASE_DIR = os.environ['GOGGLES_BASE_DIR']
assert os.path.isdir(GOGGLES_BASE_DIR)

app = Celery('goggles-celery', broker=BROKER_URL, backend=BACKEND_URL)
new_session_id = lambda: 'goggles-%s' % (time.strftime('%Y%m%dT%H%M%S'))


@app.task
def train_on_class_ids(class_ids, num_epochs=12000):
    assert type(class_ids) is list \
           and all(type(class_id) is int
                   for class_id in class_ids)

    print('Starting training for classes %s' % class_ids)

    TRAINING_SRC_PATH = os.path.join(GOGGLES_BASE_DIR, 'goggles', 'train.py')
    assert os.path.isfile(TRAINING_SRC_PATH)

    ACTIVATE_ENV_CMD = 'source activate %s' % GOGGLES_ENV
    RUN_TRAINING_CMD = 'python3 {training_src_path} with' \
                       '  filter_class_ids={filter_class_ids}' \
                       '  num_epochs={num_epochs}'.format(
        training_src_path=TRAINING_SRC_PATH,
        filter_class_ids=','.join(map(str, class_ids)),
        num_epochs=num_epochs)

    server = libtmux.Server()
    hostname = socket.gethostname()
    session, session_id = None, None
    while True:
        session_id = new_session_id()
        try: session = server.new_session(session_id)
        except TmuxSessionExists: time.sleep(2)
        else: break
    assert session is not None

    session.attached_pane.send_keys(ACTIVATE_ENV_CMD)
    time.sleep(5)
    session.attached_pane.send_keys('which python')
    session.attached_pane.send_keys(RUN_TRAINING_CMD)

    return hostname, session_id


if __name__ == '__main__':
    class_ids = list(map(int, sys.argv[1:]))
    result = app.send_task(
        'goggles-celery'
        '.train_on_class_ids',
        args=(class_ids,))

    start, timeout = time.time(), 10
    while not result.ready():
        now = time.time()
        if now - start > timeout: break
        else: time.sleep(0.2)
    print('::'.join(result.get()))
