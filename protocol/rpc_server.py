import io
import threading
from time import time

import numpy as np

from protocol import communication_pb2 as pb2
from protocol import communication_pb2_grpc as grpc
from protocol.util import serialize_np, parse_np


def choose_randomly(arr, p):
    for elem in arr:
        if np.random.random_sample() <= p:
            yield elem


class Server:
    def __init__(self, initial_model, args):
        self.initial_model = initial_model
        self.args = args
        self.client_list = []
        self.weights = []
        self.total_weight = 0.0

        self.should_contribute_list = []
        self.contributors = []
        self.last_accs = []
        self.finished = False

        self.update_list = []
        self.updates = []
        self.update = None

        self.lock = threading.RLock()
        self.cv = threading.Condition()
        self.register_wait_event = Event()
        self.should_contribute_event = Event()
        self.get_global_update_event = Event()

    def run(self):
        with self.lock:
            max_weight = np.max(self.weights)
            self.weights = [w / max_weight for w in self.weights]
            self.total_weight = np.sum(self.weights)
            print("Server initialized weights to:", self.total_weight, self.weights)
        self.register_wait_event.notify()
        while True:
            with self.cv:
                while len(self.should_contribute_list) != len(self.client_list):
                    self.cv.wait()
            with self.lock:
                acc = np.mean(self.last_accs)
                print("Round accuracy: {accuracy:.4f}%".format(accuracy=100 * acc))
                if acc > self.args['target_acc']:
                    self.finished = True
                else:
                    self.contributors = list(choose_randomly(self.should_contribute_list, self.args['q']))
                    print(f"Chosen clients for round: [{', '.join([str(x) for x in self.contributors])}]")
                    self.should_contribute_list = []
                    self.last_accs = []
            self.should_contribute_event.notify()
            if self.finished:
                print("Target accuracy reached")
                break
            with self.cv:
                while len(self.update_list) != len(self.contributors):
                    self.cv.wait()
            with self.lock:
                update = np.zeros_like(self.updates[0])
                for weight, local_update in zip(self.weights, self.updates):
                    update += weight * local_update
                update /= (self.args['q'] * self.total_weight)
                self.update = update
            self.update_list = []
            self.updates = []
            self.contributors = []
            self.get_global_update_event.notify()

    def add_client(self, client_id, client_data_points):
        with self.lock:
            self.client_list.append(client_id)
            self.weights.append(client_data_points)

    def add_should_contribute(self, client_id, last_acc):
        with self.lock, self.cv:
            self.should_contribute_list.append(client_id)
            self.last_accs.append(last_acc)
            self.cv.notify()

    def add_update(self, client_id, update):
        with self.lock, self.cv:
            if client_id in self.contributors:
                self.update_list.append(client_id)
                self.updates.append(update)
                self.cv.notify()
                return True
            return False


class RpcServer(grpc.ServerServicer):
    def __init__(self, server: Server):
        self.server = server

    def RegisterClient(self, request, context):
        self.server.add_client(request.client_id, request.client_data_len)
        self.server.register_wait_event.wait()
        with self.server.lock:
            yield pb2.RegisterResponse(
                weight=self.server.weights[request.client_id],
                total_weight=self.server.total_weight,
                model=serialize_np(self.server.initial_model)
            )

    def ShouldContribute(self, request, context):
        self.server.add_should_contribute(request.client_id, request.last_acc)
        self.server.should_contribute_event.wait()
        with self.server.lock:
            if self.server.finished:
                yield pb2.ShouldContributeResponse(contribute=False, finished=True)
            else:
                yield pb2.ShouldContributeResponse(contribute=request.client_id in self.server.contributors,
                                                   finished=False)

    def CommitUpdate(self, request, context):
        committed = self.server.add_update(request.client_id, parse_np(request.model))
        return pb2.Ack(result=committed)

    def GetGlobalUpdate(self, request, context):
        self.server.get_global_update_event.wait()
        with self.server.lock:
            yield serialize_np(self.server.update)


class Event:
    def __init__(self):
        self._evt = threading.Event()

    def wait(self):
        self._evt.wait()

    def notify(self):
        self._evt.set()
        self._evt.clear()
