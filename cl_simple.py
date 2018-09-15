from cl_base import *
from scipy import misc
from PIL import Image
import numpy as np

class CLNN_Simple(CLNN_Base):
    def __init__(self, ctx, model):
        super(CLNN_Simple, self).__init__(ctx, model)

    @staticmethod
    def load_image(file):
        return Image.open(file).convert("YCbCr")

    @staticmethod
    def resize_image(im, shape):
        if isinstance(shape, (list, tuple)) and len(shape) == 2:
            return im.resize((shape[0], shape[1]), resample=Image.BICUBIC)
        else:
            if isinstance(shape, (list, tuple)):
                shape = shape[0]
            if isinstance(shape, (float, int)):
                height = int(im.size[1] * float(shape) / im.size[0])
                return im.resize((shape, height), resample=Image.BICUBIC)

    def process_image(self, im, process_callback=None):
        im = misc.fromimage(im)
        luma = im[:, :, 0]
        in_plane = np.clip(luma.astype("float32") / 255.0, 0, 1)
        out_y = self.filter_image(in_plane, process_callback)
        luma = (np.clip(np.nan_to_num(out_y), 0, 1) * 255).astype("uint8")
        im[:, :, 0] = luma
        return misc.toimage(im, mode="YCbCr")