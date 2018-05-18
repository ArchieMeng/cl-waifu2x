__kernel void convolve_many(
            const unsigned int bx,
            const unsigned int by,
            const unsigned int num_inputs,
            const unsigned int num_outputs,
            global const float *in,
            global const float *k,
            global const float *bias,
            global float *out)
        {
            int bs = bx*by;
            int xy = get_global_id(0) + get_global_id(1) * bx;
            int z = get_global_id(2);

            float acc = 0;
            int k_off = z * num_inputs * 9;
            k += k_off;
            for (int i = 0; i < num_inputs; i++) {
                int i_off = xy + i*bs;

                for (int channel = 0;channel < 9;channel++) {
                    acc += in[i_off + (channel / 3) * bx + channel % 3] * (*(k++));
                }
            }
            acc += bias[z];
            out[xy + z*bs] = acc - 0.9 * fmin(acc, 0.f);
        }