from omegaconf import DictConfig, OmegaConf
from torch import device, nn
from torchvision.models.vgg import vgg16
from torchvision.models import resnet18
from .modules.TwoLayerConvNet.twoLayerConvNet import TwoLayerConvNet
from .modules.Submodules import Identity
from .modules.Embedding_model import EmbeddingModel
from .modules.Arcface.arcface_layer import ArcMarginProduct
from .modules.Cosface.cosface_layer import CosMarginProduct
from .modules.SoftMaxEncoder.softmax_layer import SoftMaxLayer


def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.kaiming_normal_(m.weight)
        nn.init.zeros_(m.bias)
    if isinstance(m, nn.Conv2d):
      nn.init.kaiming_normal_(m.weight)
      nn.init.zeros_(m.bias)

def get_model(cfg: DictConfig, device=device('cpu')):
    if cfg.name == "tinny model":
        model = TwoLayerConvNet(**cfg.params)
    if cfg.name == "embedding_model":
        backbone, backbone_outsize = load_backbone_modle(cfg.backbone, device=device)
        loss_fun = get_loss_layer(cfg.lossfunction,
                                  emb_size=cfg.params.emb_size,
                                  labels_num=cfg.params.labels_num,
                                  device=device,
                                  )
        fc_unit =  nn.Sequential(
            nn.Dropout(cfg.params.dropout),
            nn.Linear(backbone_outsize, cfg.params.emb_size),
            nn.BatchNorm1d(cfg.params.emb_size),
        )
        
        model = EmbeddingModel(backbone, loss_fun, fc_unit)
    return model

def load_backbone_modle(cfg: DictConfig, device=device('cpu')):
    '''
    Метод загружающий преобученные модели из torchvision
    а тагже удаляющий из них последний локальный слой для более удобного файнтюнинга
    '''
    if cfg.name == "vgg16":
        model = vgg16(cfg.pretrained)
        model.to(device)
        outsize = model.classifier[0].in_features
        model.classifier = Identity()
    elif cfg.name == "resnet18":
        model = resnet18(cfg.pretrained)
        model.to(device)
        outsize = model.fc.in_features
        model.fc = Identity()
    else:
        print("Unknow model type " + cfg.name)
        return 
    
    #Замораживаем все слои с весами
    weighted_layers = []
    for layer in list(model.modules()):
        if type(layer) is nn.Conv2d:
            weighted_layers.append(layer)
    if cfg.freeze_up:
        freeze_up_to = cfg.freeze_up_to
        if (freeze_up_to == 0):
            freeze_up_to = len(weighted_layers)
        for child in weighted_layers[:freeze_up_to]:
            for p in child.parameters():
                p.requires_grad = False
    return model, outsize


def get_loss_layer(cfg: DictConfig,emb_size, labels_num,  device=device('cpu')):
    '''
    Метод возвращающий реализацию фыбранной лосс функции
    '''
    if cfg.name == "arcface":
        loss_layer = ArcMarginProduct(emb_size=emb_size,
                                 out_feature = labels_num,
                                 s=cfg.s,
                                 m=cfg.m,
                                 auto_s=cfg.auto_s,
                                 device=device)
    elif cfg.name == "cosface":
        loss_layer = CosMarginProduct(emb_size=emb_size,
                                 out_feature = labels_num,
                                 s=cfg.s,
                                 m=cfg.m,
                                 device=device)
    elif cfg.name == "softmax":
        loss_layer = SoftMaxLayer(s=cfg.s,
                                 emb_size=emb_size,
                                 labels_num = labels_num)
    else:
        print("Unknow layer type " + cfg.name)
        return
    return loss_layer
