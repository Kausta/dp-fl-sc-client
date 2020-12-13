import threading

from protocol import communication_pb2 as pb2
from protocol import communication_pb2_grpc as grpc
from protocol.util import serialize_np, parse_np
from server import Server


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
                model=serialize_np(self.server.initial_model),
                method=self.server.method
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
