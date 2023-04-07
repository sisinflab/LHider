import smtplib, ssl
from email.message import EmailMessage
import os

NOTIFIER_FOLDER = './email_notifier'

class EmailNotifier:
    def __init__(self, config_path, message_path):
        self.config_path = config_path

        self.message_path = message_path
        self.port = 465  # for SSL
        self.server = "smtp.gmail.com"
        pass

    def read_configuration(self, path):
        senders = []
        receivers = []
        with open(path, 'r') as file:
            for row in file.readlines():
                tokens = row.split('\t')
                role = tokens[0]
                if role == 'receiver':
                    receivers.append(tokens[1].replace('\n', ''))
                elif role == 'sender':
                    senders.append({
                        'mail': tokens[1],
                        'pass': tokens[2].replace('\n', '')
                    })
        return senders, receivers


DATA_PATH = os.path.join(NOTIFIER_FOLDER, './info.txt')
MESSAGE_PATH = os.path.join(NOTIFIER_FOLDER, 'message.txt')

PORT = 465  # For SSL
SMPT_SERVER = "smtp.gmail.com"


def read_data(path):
    senders = []
    receivers = []
    with open(path, 'r') as file:
        for row in file.readlines():
            tokens = row.split('\t')
            role = tokens[0]
            if role == 'receiver':
                receivers.append(tokens[1].replace('\n', ''))
            elif role == 'sender':
                senders.append({
                    'mail': tokens[1],
                    'pass': tokens[2].replace('\n', '')
                })
    return senders, receivers


def read_message(path):
    messages = []
    with open(path, 'r') as file:
        for row in file.readlines():
            tokens = row.split('\t')
            role = tokens[0]
            messages.append({
                'role': role,
                'subject': tokens[1],
                'text': tokens[2].replace('\n', '')
            })
    return messages


def send_email(senders=None, receivers=None, messages=None, error=False, error_exception=None):

    if isinstance(senders, list):
        pass
    else:
        senders_path = os.path.abspath(DATA_PATH)

        if isinstance(senders, str):
            senders_path = os.path.abspath(os.path.join(NOTIFIER_FOLDER, senders))
            assert os.path.exists(senders_path), f'senders file path at \'{senders_path}\' does not exist'

        senders, _ = read_data(senders_path)

    if isinstance(receivers, list):
        pass
    else:
        receivers_path = os.path.abspath(DATA_PATH)

        if isinstance(receivers, str):
            receivers_path = os.path.abspath(os.path.join(NOTIFIER_FOLDER, receivers))
            assert os.path.exists(receivers_path), 'receivers file path at \'{receivers_path}\' does not exist'

        _, receivers = read_data(receivers_path)

    if isinstance(messages, list):
        pass
    else:
        messages_path = os.path.abspath(MESSAGE_PATH)

        if isinstance(messages, str):
            messages_path = os.path.abspath(os.path.join(NOTIFIER_FOLDER, messages))
            assert os.path.exists(messages_path), f'messages file path at \'{messages_path}\' does not exist'

        messages = read_message(messages_path)

    for sender in senders:
        sender_mail = sender['mail']
        sender_pass = sender['pass']

        for receiver in receivers:

            for message in messages:
                role = message['role']
                if error:
                    if role == 'error':
                        if error_exception:
                            message['text'] = message['text'] + f'\nException: {error_exception}'
                        send(sender_mail, receiver, message, sender_pass)
                else:
                    if role == 'all' or role == receiver:
                        send(sender_mail, receiver, message, sender_pass)


def send(sender, receiver, message, password, smpt_server=SMPT_SERVER, port=PORT):
    print(f'sending message from {sender} to {receiver}')
    context = ssl.create_default_context()

    message_obj = EmailMessage()
    message_obj['Subject'] = message['subject']
    message_obj['From'] = sender
    message_obj['To'] = receiver
    message_obj.set_content(message['text'])

    try:
        with smtplib.SMTP_SSL(smpt_server, port, context=context) as server:
            server.login(sender, password)
            server.send_message(message_obj)
        print(f'email from {sender} to {receiver} sent')
    except:
        print(f'an error occurred while sending an email from {sender} to {receiver} sent')


def run_with_mail(func):
    pass
