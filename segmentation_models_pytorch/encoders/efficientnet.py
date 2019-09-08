from efficientnet_pytorch import EfficientNet, get_model_params
from efficientnet_pytorch.utils import url_map
from torch.utils import model_zoo


class EfficientNetEncoder(EfficientNet):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.pretrained = True
        del self._fc

    def forward(self, inputs):
        return self.extract_features(inputs)

    def load_state_dict(self, state_dict, **kwargs):
        model_name = 'efficientnet-b1'
        state_dict = model_zoo.load_url(url_map[model_name])
        #if load_fc:
        #    model.load_state_dict(state_dict)
        #else:
        state_dict.pop('_fc.weight')
        state_dict.pop('_fc.bias')
        res = super().load_state_dict(state_dict, strict=False)
        #assert str(res.missing_keys) == str(['_fc.weight', '_fc.bias']), 'issue loading pretrained weights'
        print('Loaded pretrained weights for {}'.format(model_name))


efficientnet_encoders = {
    'efficientnetb1': {
        'encoder': EfficientNetEncoder,
        'pretrained_settings': {
            'imagenet': {
                'url': 'http://storage.googleapis.com/public-models/efficientnet/efficientnet-b1-f1951068.pth',
                'input_space': 'RGB',
                'input_size': [3, 224, 224],
                'input_range': [0, 1],
                'mean': [0.485, 0.456, 0.406],
                'std': [0.229, 0.224, 0.225],
                'num_classes': 1000
            },
        },
        'out_shapes': (592, 296, 152, 80, 32),
        'params': {
            'blocks_args': get_model_params('efficientnet-b1', None)[0],
            'global_params': get_model_params('efficientnet-b1', None)[1]
        },
    }
}
