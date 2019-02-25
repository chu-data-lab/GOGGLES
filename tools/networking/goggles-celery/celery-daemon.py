import os
import random
import socket
import string
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
app.conf.update(worker_prefetch_multiplier=1)

new_session_id = lambda: 'goggles-%s-%s' % (
    time.strftime('%Y%m%dT%H%M%S'),
    ''.join(random.sample(string.ascii_lowercase, 4)))


@app.task(bind=True)
def run_script(self, script_path, script_args):
    print(f'Running {script_path} with "{script_args}"')

    lock_file = None
    lock_dir = os.path.join(
        GOGGLES_BASE_DIR, '_scratch', 'locks')
    if not os.path.exists(lock_dir):
        os.makedirs(lock_dir)

    script_full_path = os.path.join(GOGGLES_BASE_DIR, script_path)
    assert os.path.isfile(script_full_path)

    activate_env_cmd = 'source activate %s' % GOGGLES_ENV
    run_script_cmd = f'python3 {script_full_path} {script_args}'

    server = libtmux.Server()
    hostname = socket.gethostname()
    session, session_id = None, None
    while True:
        session_id = new_session_id()
        try:
            session = server.new_session(session_id)

            lock_file = os.path.join(lock_dir, '%s.lock' % session_id)
            with open(lock_file, 'w') as f:
                f.write(self.request.id)

        except TmuxSessionExists:
            time.sleep(2)

        else:
            break

    assert session is not None
    assert os.path.isfile(lock_file)

    session.attached_pane.send_keys(activate_env_cmd)
    time.sleep(4)  # Waiting for env to get activated
    session.attached_pane.send_keys(
        f'{run_script_cmd};'
        f'rm -f {lock_file}; '
        f'exit')

    while os.path.exists(lock_file):
        self.update_state(
            state='RUNNING',
            meta={'hostname': hostname,
                  'session_id': session_id})

        time.sleep(20)

    self.update_state(
        state='FINISHED',
        meta={'hostname': hostname,
              'session_id': session_id})

    return hostname, session_id
