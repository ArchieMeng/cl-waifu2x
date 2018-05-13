#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
import pyopencl as cl
import time, sys
from scipy import misc
from PIL import Image
import os

dirname = os.path.dirname(os.path.realpath(__file__))
sys.path.append(dirname)
from cl_simple import CLNN_Simple
import argparse

arg_parser = argparse.ArgumentParser()
arg_parser.add_argument('--input', '-i', default='sample.jpg')
arg_parser.add_argument('--output', '-o', default=None)
arg_parser.add_argument('--model_file', '-m', default=os.path.join(dirname, 'models/scale2.0x_model.json'))
args = arg_parser.parse_args()

infile = args.input
model_path = args.model_file
model_name = os.path.basename(model_path)
default_outfile_name = '.'.join(infile.split('.')[:-1]) + '(' + model_name + ')'
outfile = args.output or default_outfile_name + '.' + infile.split('.')[-1]

scale = "scale" in model_path

# ctx = cl.create_some_context(interactive=True)
ctx = cl.Context(
    dev_type=cl.device_type.ALL,
    properties=[(cl.context_properties.PLATFORM, cl.get_platforms()[1])]
)
print(ctx)
nn = CLNN_Simple(ctx, model_path)

im = Image.open(infile).convert("YCbCr")

if scale:
    im = im.resize((2*im.size[0], 2*im.size[1]), resample=Image.BICUBIC)

im = misc.fromimage(im)
luma = im[:,:,0]
#misc.toimage(luma, mode="L").save("luma_"+outfile)

in_plane = np.clip(luma.astype("float32") / 255.0, 0, 1)


def progress(frac):
    sys.stderr.write("\r%.1f%%..." % (100 * frac))
o_np = nn.filter_image(in_plane, progress)
sys.stderr.write("Done\n")
sys.stderr.write("%d pixels/sec\n" % nn.pixels_per_second)
sys.stderr.write("%f Gflops/sec\n" % (nn.ops_per_second / (10. ** 9)))

luma = (np.clip(np.nan_to_num(o_np), 0, 1) * 255).astype("uint8")
#misc.toimage(luma, mode="L").save("luma_o_"+outfile)
im[:,:,0] = luma
misc.toimage(im, mode="YCbCr").convert("RGB").save(outfile)

