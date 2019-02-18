import os

import pushed


APP_KEY = os.getenv('GOGGLES-PUSHED-APP-KEY')
APP_SECRET = os.getenv('GOGGLES-PUSHED-APP-SECRET')


def notify(message, namespace=None):
    # best-effort notifier

    try:
        if namespace is not None:
            message = f'[{namespace}] ' + message

        p = pushed.Pushed(APP_KEY, APP_SECRET)
        shipment = p.push_app(message)
    except:
        pass
