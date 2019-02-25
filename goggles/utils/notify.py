import os

from slackclient import SlackClient


API_TOKEN = os.getenv('GOGGLES_SLACK_API_TOKEN')


def notify(message, namespace=None):
    # best-effort notifier

    try:
        if namespace is not None:
            message = f'[{namespace}] ' + message

        sc = SlackClient(API_TOKEN)
        sc.api_call(
            'chat.postMessage',
            channel='goggles-experiments',
            text=message,
            username='goggles-bot')

    except:
        pass
