from configs.mnist import two_stage_mnist_configs


def get_config():
    config = two_stage_mnist_configs.get_config()
    config.model.transition_steps = 0
    config.model.stage1_prediction = "epsilon"
    return config
