from timm import models

model = models.efficientnet_b3()
setattr(model.conv_stem, "in_channels", 1)
setattr(model.classifier, "out_features", 5)

print(model)