from configs.mnist import two_stage_mnist_light_prefix_configs


def get_config():
    config = two_stage_mnist_light_prefix_configs.get_config()
    config.model.prediction_target = "x0"
    return config
