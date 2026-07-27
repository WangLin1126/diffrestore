from configs.ffhq import two_stage_ffhq_light_prefix_128_configs


def get_config():
    config = two_stage_ffhq_light_prefix_128_configs.get_config()
    config.model.prediction_target = "x0"
    return config
