__all__ = ['build_head']

def build_head(config):
    from .rec_abinet_head import ABIHead
    support_dict = [
        'ABIHead'
    ]

    module_name = config.pop('name')
    assert module_name in support_dict, Exception('head only support {}'.format(
        support_dict))
    module_class = eval(module_name)(**config)
    return module_class

if __name__=='__main__':
    import yaml
    with open('configs/rec/rec_r45_abinet.yml', 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)['Architecture']['Head']
    config['in_channels'] = 512
    config['out_channels'] = 37
    build_head(config)