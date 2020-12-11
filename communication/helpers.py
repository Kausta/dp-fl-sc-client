from communication import communication_pb2
import io
import numpy as np


# Returns a dictionary.
def parse_public_parameters(parameter_stream: iter):
    params = {}
    for x in parameter_stream:
        value = x.value_int if x.value_int >= 0 else x.value_float if x.value_float >= 0 else x.value_string
        params[x.name] = value
    return params


# Returns an iterator.
def serialize_public_parameters(parameters: dict):
    for k, v in parameters.items():
        if isinstance(v, int):
            yield communication_pb2.PublicParameter(name=k, value_int=v, value_float=-1, value_string="")
        elif isinstance(v, float):
            yield communication_pb2.PublicParameter(name=k, value_int=-1, value_float=v, value_string="")
        elif isinstance(v, str):
            yield communication_pb2.PublicParameter(name=k, value_int=-1, value_float=-1, value_string=v)


# Returns an iterator.
def serialize_contributions(contributions: dict, self_id: int):
    return (communication_pb2.Contribution(target_id=target_id, contribution=contribution, contributor_id=self_id) \
            for target_id, contribution in contributions.items())


def serialize_model(model, weight: int):
    file = io.BytesIO()
    np.savez_compressed(file, model=model, weight=weight)
    data = file.getbuffer()
    return communication_pb2.Model(data = data.tobytes())


def parse_model(model: communication_pb2.Model):
    file = io.BytesIO()
    file.write(model.data)
    file.seek(0)
    content = np.load(file)
    return content['model'], content['weight'].item()