from communication import communication_pb2

# Returns a dictionary.
def parse_public_parameters(parameter_stream: iter):
    return {x.name: x.value for x in parameter_stream}

# Returns an iterator.
def serialize_public_parameters(parameters: dict):
    return (communication_pb2.PublicParameter(name=k, value=v) for k, v in parameters.items())