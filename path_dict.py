import socket
import os

class PathDict (object):
    def __init__ (self):
        hostname = socket.gethostname()
        if 'abci' in hostname:
            self.proj_root = os.path.dirname(os.path.abspath(__file__))
        else:
            self.proj_root = os.path.dirname(os.path.abspath(__file__))
