import yaml, paddle, os, glob

from PIL import Image
from modules.utils import preprocess
from modules.model_abinet_iter import ABINetIterModel

device = 'cpu'
device = paddle.set_device(device)
with open('./configs/abinet.yml', 'r') as f:
    cfg = yaml.load(f, Loader=yaml.FullLoader)
model = ABINetIterModel(cfg)
state = paddle.load('pretrain_models/pretrain_vm.pdparams')
model.vision.load_dict(state)
state = paddle.load('pretrain_models/pretrain_lm.pdparams')
model.language.load_dict(state)
state = paddle.load('pretrain_models/pretrain_alignment.pdparams')
model.alignment.load_dict(state)
model.eval()

paths = [os.path.join('./figs/test', fname) for fname in os.listdir('./figs/test')]
paths = sorted(paths)
for im_path in paths:
    img = Image.open(im_path).convert('RGB')
    img = paddle.to_tensor(preprocess(img, 128 ,32)).unsqueeze(0)
    out, _, _ = model(img)
    pt_text, pt_scores, pt_lengths = model._get_text(out['logits'])
    print(f'{im_path}: {pt_text[0]}')