from configs.ffhq import two_stage_ffhq_light_prefix_configs


def get_config():
    config = two_stage_ffhq_light_prefix_configs.get_config()
    config.model.K_noise = 40
    config.model.K_blur = 60
    config.model.K = config.model.K_noise + config.model.K_blur
    config.model.boundary_noise_sigma = 0.25
    config.model.sequential_blur = True
    config.model.noise_schedule = two_stage_ffhq_light_prefix_configs._ddpm_linear_noise_schedule(
        config.model.boundary_noise_sigma, config.model.K_noise)
    positive_blur_schedule = two_stage_ffhq_light_prefix_configs.np.exp(
        two_stage_ffhq_light_prefix_configs.np.linspace(
            two_stage_ffhq_light_prefix_configs.np.log(config.model.blur_sigma_min),
            two_stage_ffhq_light_prefix_configs.np.log(config.model.blur_sigma_max),
            config.model.K_blur
        )
    )
    config.model.blur_schedule = two_stage_ffhq_light_prefix_configs.np.concatenate(
        [[0.0], positive_blur_schedule]
    )
    config.model.denoise_sampling_stds = two_stage_ffhq_light_prefix_configs._denoise_sampling_stds(
        config.model.noise_schedule)
    return config
