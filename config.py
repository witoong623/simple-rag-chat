import yaml


class Config:
    openai_base_url: str | None
    openai_api_key: str | None

    def __init__(self, config_filepath: str):
        with open(config_filepath) as f:
            config_obj = yaml.safe_load(f)

        self.openai_base_url = config_obj.get('openai_compatible_base_url')
        self.openai_api_key = config_obj.get('openai_api_key')

        assert self.openai_base_url or self.openai_base_url, "openai_base_url or openai_api_key is required."
