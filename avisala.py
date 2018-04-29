
import keras
import socket
import json

class Avisala_Callback(keras.callbacks.Callback):
    def __init__(self, receivers,
        host = '127.0.0.1',
        port = 20081,
        notify_train_begin = True, notify_train_end = True,
        notify_epoch_begin = False, notify_epoch_end = True,
        notify_batch_begin = False, notify_batch_end = False):

        self.host = host
        self.port = port
        self.notify_train_begin = notify_train_begin
        self.notify_train_end = notify_train_end
        self.notify_epoch_begin = notify_epoch_begin
        self.notify_epoch_end = notify_epoch_end
        self.notify_batch_begin = notify_batch_begin
        self.notify_batch_end = notify_batch_end

    def get_connection(self):
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.connect((self.host, self.port))
        return s

    def send_dict(self, s, dict):
        str_content = json.dumps(dict)
        content = str_content.encode()
        s.send(content)
        return


    def on_train_begin(self, logs={}):
        self.logs = []
        if self.notify_train_begin:
            notification = dict()
            notification['device'] = 'HOHNE_IOS'
            notification['model'] = self.model.name
            notification['event'] = 'on_train_begin'
            notification['train_loss'] = ''
            s = self.get_connection()
            self.send_dict(s, notification)
            s.close()
        return

    def on_train_end(self, logs={}):
        self.logs.append(logs)
        if self.notify_train_end:
            notification = dict()
            notification['device'] = 'HOHNE_IOS'
            notification['model'] = self.model.name
            notification['event'] = 'on_train_end'
            notification['train_loss'] = str(logs.get('loss'))
            s = self.get_connection()
            self.send_dict(s, notification)
            s.close()
        return

    def on_epoch_begin(self, epoch, logs={}):
        self.logs.append(logs)
        if self.notify_epoch_begin:
            notification = dict()
            notification['device'] = 'HOHNE_IOS'
            notification['model'] = self.model.name
            notification['event'] = 'on_epoch_begin'
            notification['epoch'] = str(epoch)
            notification['train_loss'] = str(logs.get('loss'))
            notification['validation_loss'] = str(logs.get('val_loss'))
            s = self.get_connection()
            self.send_dict(s, notification)
            s.close()
        return

    def on_epoch_end(self, epoch, logs={}):
        self.logs.append(logs)
        if self.notify_epoch_end:
            notification = dict()
            notification['device'] = 'HOHNE_IOS'
            notification['model'] = self.model.name
            notification['event'] = 'on_epoch_end'
            notification['epoch'] = str(epoch)
            notification['train_loss'] = str(logs.get('loss'))
            notification['validation_loss'] = str(logs.get('val_loss'))
            s = self.get_connection()
            self.send_dict(s, notification)
            s.close()
        return

    def on_batch_begin(self, batch, logs={}):
        self.logs.append(logs)
        if self.notify_batch_begin:
            notification = dict()
            notification['device'] = 'HOHNE_IOS'
            notification['model'] = self.model.name
            notification['event'] = 'on_batch_begin'
            notification['train_loss'] = str(logs.get('loss'))
            s = self.get_connection()
            self.send_dict(s, notification)
            s.close()
        return

    def on_batch_end(self, batch, logs={}):
        self.logs.append(logs)
        if self.notify_batch_end:
            notification = dict()
            notification['device'] = 'HOHNE_IOS'
            notification['model'] = self.model.name
            notification['event'] = 'on_batch_begin'
            notification['train_loss'] = str(logs.get('loss'))
            s = self.get_connection()
            self.send_dict(s, notification)
            s.close()
        return
