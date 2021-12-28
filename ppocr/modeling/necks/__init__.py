__all__ = ['build_neck']

def build_neck(config):
    from .transformer import ABINeck
    support_dict = ['ABINeck']

    module_name = config.pop('name')
    assert module_name in support_dict, Exception('neck only support {}'.format(
        support_dict))
    module_class = eval(module_name)(**config)
    return module_class

if __name__=='__main__':
    import yaml
    with open('configs/rec/rec_r45_abinet.yml', 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)['Architecture']['Neck']
    build_neck(config)