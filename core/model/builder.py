import copy
from utils import MODEL_REGISTRY
from .swin_transformer_seq import swint3d_tiny
# from torchvision.models import resnet50

# from .bert_model import BERT
# from .multibert_model import MultiBERT
# from .gru_model import BiGRUEncoder

def build_model(cfg):
    # model_cfg = copy.deepcopy(cfg)
    # try:
    #     model_cfg = model_cfg['model']
    # except Exception:
    #     raise 'should contain {model}'

    # model = MODEL_REGISTRY.get(model_cfg['name'])(**model_cfg['args'])
    model  = swint3d_tiny(
        # pretrained="/mmaction2/Emo-Swin/swin_base_patch4_window7_224_22k.pth"
        pretrained = "/mmaction2/Emo-Swin/swin_base_patch244_window877_kinetics400_22k.pth"
    )
    # model = resnet50(pretrained=True)
    # model.fc = nn.Linear(512,8)

    return model
