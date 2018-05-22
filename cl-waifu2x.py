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
default_model = os.path.join(dirname, 'models/scale2.0x_model.json')
arg_parser.add_argument(
    '--input',
    '-i',
    default='sample.jpg'
)
arg_parser.add_argument(
    '--output',
    '-o',
    default=None
)
arg_parser.add_argument(
    '--model_file',
    '-m',
    nargs='+',
    type=str,
    default=[default_model]
)
arg_parser.add_argument(
    '--resize',
    '-r',
    nargs='+',
    type=int,
    default=[],
    metavar="height [width]",
    help="""set output image size to (height * width)"""
)
args = arg_parser.parse_args()

infile = args.input
model_paths = args.model_file
model_names = [os.path.basename(model_path) for model_path in model_paths]
default_outfile_name = '.'.join(infile.split('.')[:-1]) + '(' + '+'.join(model_names) + ')'
outfile = args.output or default_outfile_name + '.' + infile.split('.')[-1]

ctx = cl.create_some_context(interactive=True)
# ctx = cl.Context(
#     devices=[cl.get_platforms()[1].get_devices()[0], cl.get_platforms()[1].get_devices()[1]],
#     properties=[(cl.context_properties.PLATFORM, cl.get_platforms()[1])]
# )
print(ctx)

im = CLNN_Simple.load_image(infile)
need_resize = args.resize and len(args.resize) <= 2
if need_resize:
    im = CLNN_Simple.resize_image(im, args.resize)

for model_path in model_paths:
    nn = CLNN_Simple(ctx, model_path)

    scale = "scale" in model_path

    # need resize
    if (not need_resize) and scale:
        im = im.resize((2*im.size[0], 2*im.size[1]), resample=Image.BICUBIC)



    def progress(frac):
        sys.stderr.write("\r%.1f%%..." % (100 * frac))
    im = nn.process_image(im, progress)
    sys.stderr.write("Done\n")
    sys.stderr.write("%d pixels/sec\n" % nn.pixels_per_second)
    sys.stderr.write("%f Gflops/sec\n" % (nn.ops_per_second / (10. ** 9)))
    # misc.toimage(luma, mode="L").save("luma_o_"+outfile)

im.convert('RGB').save(outfile)
print("output to:", outfile)

