# cl-waifu2x

**WARNING: This software is under active development and not yet inteded for
production or use by end-users. It is not yet optimized. Tread with caution.**

cl-waifu2x is an implementation of the waifu2x algorithm in OpenCL. It focuses
on use of the neural network algorithm, not its training, and therefore it
relies on models produced with the original waifu2x.

cl-waifu2x aims to be compatible with most mainstream OpenCL implementations,
including GPU-based and CPU-based ones from major vendors.

Based on [waifu2x by nagadomi](https://github.com/nagadomi/waifu2x).

## Dependencies

* Python 2.7 or Python3
* numpy
* scipy
* PIL (or Pillow)
* PyOpenCL

And an OpenCL implementation.

## Usage
    $ python3 cl-waifu2x.py --help
    usage: cl-waifu2x.py [-h] [--input INPUT] [--output OUTPUT]
                         [--model_file MODEL_FILE [MODEL_FILE ...]]

    optional arguments:
      -h, --help            show this help message and exit
      --input INPUT, -i INPUT
      --output OUTPUT, -o OUTPUT
      --model_file MODEL_FILE [MODEL_FILE ...], -m MODEL_FILE [MODEL_FILE ...]

    You can use many models to operate the image, by this way, you can scale the images first and then decrease noise by
    given parameters like "-m models/scale2.0x_model.json models/noise1_model.json"

    $ python3 cl-waifu2x.py -i miku_small.png -o miku_small_cl.png -m qmodels/scale2.0x_model.json
    Choose platform:
    [0] <pyopencl.Platform 'Intel(R) OpenCL' at 0x7fa4d7f10110>
    [1] <pyopencl.Platform 'NVIDIA CUDA' at 0x7fa4d8026f80>
    Choice [0]:
    Set the environment variable PYOPENCL_CTX='0' to avoid being asked again.
    100.0%...Done
    39918 pixels/sec
    3.820713 Gflops/sec

OpenCL implementations that are being used for testing:
* Intel (CPU) (test platform: Intel Core i7 3820QM)
* Nvidia (GPU) (test platform: Nvidia GeForce GTX 660M)
* Intel (CPU) (test platform: Intel Core i7 7500u)
* Intel (GPU) (test platform: Intel(R) HD Graphics 620)

## Performance

The current kernel is very dumb and not yet GPU-optimized. Performance is
currently about equal on CPU and GPU, and about 6 times slower than the original
waifu2x CUDA version on the same GPU, though also several times faster than
the trivial
[single-threaded waifu2x.py](tools/waifu2x.py)
on the same CPU.
(PS: on Intel 7-th gen platform, GPU is somehow 12.5% slower than CPU.)
