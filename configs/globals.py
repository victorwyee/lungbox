"""Global config class for project-wide configs."""


class GlobalConfig:
    """Config class for globals."""

    __conf = {
        'ROOT_DIR': '/projects/lungbox',
        'AWS_REGION': 'us-west-2',
        'AWS_ACCESS_KEY': '',
        'S3_BUCKET_NAME': 'lungbox',
        'S3_CLASS_INFO_KEY': 'data/raw/stage_1_detailed_class_info.csv',
        'S3_TRAIN_BOX_KEY': 'data/raw/stage_1_train_labels.csv',
        'S3_CLASS_INFO_PATH': 's3://lungbox/data/raw/stage_1_detailed_class_info.csv',
        'S3_TRAIN_BOX_PATH': 's3://lungbox/data/raw/stage_1_train_labels.csv',
        'S3_STAGE1_TRAIN_IMAGE_DIR': 'data/raw/stage_1_train_images',
        'S3_STAGE1_TEST_IMAGE_DIR': 'data/raw/stage_1_test_images',
        'MODEL_DIR': '/projects/lungbox/models'
    }
    __setters = ['AWS_ACCESS_KEY']

    @staticmethod
    def get(name):
        """Get config by name."""
        return GlobalConfig.__conf[name]

    @staticmethod
    def set(name, value):
        """Set config if config is settable."""
        if name in GlobalConfig.__setters:
            GlobalConfig.__conf[name] = value
        else:
            raise NameError("Name not accepted in set() method")
