from models.backbone.resnet18 import ResNet18Backbone
from models.encoder.deco_encoder import DECOEncoder
from models.decoder.deco_decoder import DECODecoder
from models.deco_model import DECOModel

backbone = ResNet18Backbone(pretrained=True)
encoder  = DECOEncoder(in_channels=512, d_model=256, num_layers=3)  # (only an example)
decoder  = DECODecoder(d_model=256, num_queries=100, num_layers=3, num_classes=80)

model = DECOModel(backbone, encoder, decoder)
# model.to(device)
