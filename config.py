from grpc._cython.cygrpc import CompressionAlgorithm
from grpc._cython.cygrpc import CompressionLevel


def get_config():
    return {
        'data_dir': "data/mnist/",  # Data directory
        'key_path': 'test_key.pem',  # HE test key
        'factor_exp': 8,  # HE exponential factor for converting floats to integers
        'lr': 0.01,  # Base learning rate
        'S': 1,  # Sensitivity constant from DP-FedAvg
        'epsilon': 1,  # Epsilon required for epsilon-differential privacy
        'batch_size': 100,  # Local batch sizes
        'local_epochs': 1,  # Local epochs per round
        'target_acc': 0.70,
        'q': 1  # Fraction of clients joining in any round, 0 < q <= 1,
                #   indicates probability of a client participating to a round
    }


def grpc_options():
    MAX_MESSAGE_LENGTH = 2000000000
    return [('grpc.default_compression_algorithm', CompressionAlgorithm.gzip),
            ('grpc.grpc.default_compression_level', CompressionLevel.high),
            ('grpc.max_send_message_length', MAX_MESSAGE_LENGTH),
            ('grpc.max_receive_message_length', MAX_MESSAGE_LENGTH)]
