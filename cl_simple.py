from cl_base import *


class CLNN_Simple(CLNN_Base):
    # Todo this class receive any type and any size of images, and output images processed by CLNN_base
    def __init__(self, ctx, model):
        super(CLNN_Simple, self).__init__(ctx, model)
