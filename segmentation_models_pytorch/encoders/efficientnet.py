from efficientnet_pytorch import EfficientNet, get_model_params
from efficientnet_pytorch.utils import url_map, relu_fn
from fastai.text import Swish
from torch.utils import model_zoo
from torch.nn import *


def custom_head(in_channels, out_channels):
    return Sequential(
        Dropout(),
        Linear(in_channels, 512),
        ReLU(inplace=True),
        Dropout(),
        Linear(512, out_channels)
    )


class EfficientNetEncoder(EfficientNet):

    def __init__(self, *args, **kwargs):
        self.out_shapes = kwargs['out_shapes']
        del kwargs['out_shapes']
        super().__init__(*args, **kwargs)
        self.pretrained = True
        del self._fc

    def forward(self, inputs):
        # Stem
        x = relu_fn(self._bn0(self._conv_stem(inputs)))

        outputs = []

        # Blocks
        for idx, block in enumerate(self._blocks):
            drop_connect_rate = self._global_params.drop_connect_rate
            if drop_connect_rate:
                drop_connect_rate *= float(idx) / len(self._blocks)
            x = block(x, drop_connect_rate=drop_connect_rate)
            outputs.append(x)

        # Head
        x = relu_fn(self._bn1(self._conv_head(x)))

        result = []
        i = 0
        r = list(reversed([80, 40, 24, 16]))  # [592, 296, 152, 80, 35, 32]
        for l in outputs:
            sz = l.shape
            if i < len(r) and sz[1] == r[i]:
                result.append(l)
                i = i + 1
        return x, (*reversed(result)),

    def load_state_dict(self, state_dict, **kwargs):
        model_name = 'efficientnet-b1'
        state_dict = model_zoo.load_url(url_map[model_name])
        # if load_fc:
        #    model.load_state_dict(state_dict)
        # else:
        state_dict.pop('_fc.weight')
        state_dict.pop('_fc.bias')
        res = super().load_state_dict(state_dict, strict=False)
        # assert str(res.missing_keys) == str(['_fc.weight', '_fc.bias']), 'issue loading pretrained weights'
        print('Loaded pretrained weights for {}'.format(model_name))


efficientnet_encoders = {
    'efficientnetb0': {
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
        'out_shapes': (592, 296, 80, 35, 32), # skip connection output channels
        'params': {
            'blocks_args': get_model_params('efficientnet-b1', None)[0],
            'global_params': get_model_params('efficientnet-b1', None)[1],
            'out_shapes': (592, 296, 80, 35, 32)
        },
    }
}
