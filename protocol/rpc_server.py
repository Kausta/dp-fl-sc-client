import threading
from functools import reduce

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

    def GetSystemSize(self, request, context):
        return pb2.SystemSizeResponse(system_size=self.server.system_size)

    def ForwardNoiseContributions(self, request_iterator, context):
        # Collect the noise contributions.
        contributions = list(map(lambda noise_cont: {'contributor': noise_cont.contributor_id,
                                                     'target': noise_cont.target_id,
                                                     'contribution': noise_cont.contribution},
                                 request_iterator))
        contributor_id = contributions[0]['contributor']
        self.server.add_noise_contributions(contributor_id, contributions)
        self.server.noise_contributions_event.wait()
        with self.server.lock:
            # Flat map
            it = filter(lambda c: c['target'] == contributor_id,
                        reduce(list.__add__, self.server.noise_contributions.values()))
            for nc in it:
                yield pb2.NoiseContribution(contributor_id=nc['contributor'],
                                            target_id=nc['target'],
                                            contribution=nc['contribution'])

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
        self.server.add_wait_global_update()
        self.server.get_global_update_event.wait()
        with self.server.lock:
            yield serialize_np(self.server.update)
