from grpc._cython.cygrpc import CompressionAlgorithm
from grpc._cython.cygrpc import CompressionLevel


def get_config():
    return {
        'data_dir': "data/mnist/",
        'key_path': 'test_key.pem',
        'factor_exp': 8,
        'lr': 0.01,
        'S': 1,
        'epsilon': 1,
        'batch_size': 100,
        'local_epochs': 5,
        'target_acc': 0.70,
        'q': 1
    }


def grpc_options():
    MAX_MESSAGE_LENGTH = 2000000000
    return [('grpc.default_compression_algorithm', CompressionAlgorithm.gzip),
            ('grpc.grpc.default_compression_level', CompressionLevel.high),
            ('grpc.max_send_message_length', MAX_MESSAGE_LENGTH),
            ('grpc.max_receive_message_length', MAX_MESSAGE_LENGTH)]
