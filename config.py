def get_config():
    return {
        'data_dir': "data/mnist/",
        'key_path': 'test_key.pem',
        'factor_exp': 16,
        'lr': 0.01,
        'S': 1,
        'epsilon': 1,
        'batch_size': 100,
        'local_epochs': 5,
        'target_acc': 0.70,
        'q': 1
    }
