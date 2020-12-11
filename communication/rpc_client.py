from communication import communication_pb2, helpers
from communication import communication_pb2_grpc
from client import Client


class RpcClient(communication_pb2_grpc.ClientServicer):
    client = None

    def __init__(self, client: Client):
        self.client = client

    def NewClient(self, request, ctx):
        self.client.new_client(request.client_id, request.client_address)
        return communication_pb2.Ack(result=1)

    def Initialize(self, request, ctx):
        self.client.initialize(helpers.parse_public_parameters(request))
        return communication_pb2.Ack(result=1)

    def InitializeModel(self, request, ctx):
        init_model, _ = helpers.parse_model(request)
        self.client.initialize_model(init_model)
        return communication_pb2.Ack(result=1)

    def ClientUpdate(self, request, ctx):
        updated_model, weight = self.client.client_update()
        return helpers.serialize_model(updated_model, weight)

    def ClientApplyUpdate(self, request, context):
        updated_model, _ = helpers.parse_model(request)
        self.client.client_apply_update(updated_model)
        return communication_pb2.Ack(result=1)

    def ReceiveContribution(self, request, ctx):
        contributor_id = request.contributor_id
        contribution = request.contribution
        self.client.receive_contribution(contributor_id, contribution)
        return communication_pb2.Ack(result=1)

    def Setup(self, request, ctx):
        self.client.setup()
        return communication_pb2.Ack(result=1)
