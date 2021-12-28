__all__ = ["build_backbone"]

def build_backbone(config, model_type):
    if model_type == "rec":
        from .rec_resnet_45 import ResNet45
        support_dict = ['ResNet45']

    module_name = config.pop("name")
    assert module_name in support_dict, Exception(
        "when model typs is {}, backbone only support {}".format(model_type,
                                                                 support_dict))
    module_class = eval(module_name)(**config)
    return module_class
