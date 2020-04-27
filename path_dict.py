import socket
import os

class PathDict (object):
    def __init__ (self):
        hostname = socket.gethostname()
        if 'abci' in hostname:
            self.surgery_ds_root = '/groups1/gcb50205/lzq/dataset/JIGSAWS'
            self.proj_root = os.path.dirname(os.path.abspath(__file__))
            print(self.proj_root)
        else:
            # self.surgery_ds_root = '/home/shinkyo/dataset/JIGSAWS'
            # self.proj_root = '/home/shinkyo/lzq/ModelVisualization'
            self.proj_root = os.path.dirname(os.path.abspath(__file__))
