import io
import numpy as np
from protocol import communication_pb2 as pb2


def serialize_np(model):
    file = io.BytesIO()
    np.save(file, model)
    data = file.getbuffer()
    return pb2.Model(data=data.tobytes())


def parse_np(model: pb2.Model):
    file = io.BytesIO()
    file.write(model.data)
    file.seek(0)
    content = np.load(file, allow_pickle=True)
    return content
