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
            for (int i = 0; i < num_inputs; i++) {
                int i_off = xy + i*bs;
                int ki = 9 * i;
                acc += in[i_off + 0*bx + 0] * k[k_off + ki + 0];
                acc += in[i_off + 0*bx + 1] * k[k_off + ki + 1];
                acc += in[i_off + 0*bx + 2] * k[k_off + ki + 2];
                acc += in[i_off + 1*bx + 0] * k[k_off + ki + 3];
                acc += in[i_off + 1*bx + 1] * k[k_off + ki + 4];
                acc += in[i_off + 1*bx + 2] * k[k_off + ki + 5];
                acc += in[i_off + 2*bx + 0] * k[k_off + ki + 6];
                acc += in[i_off + 2*bx + 1] * k[k_off + ki + 7];
                acc += in[i_off+ 2*bx + 2] * k[k_off + ki + 8];
            }
            acc += bias[z];
            out[xy + z*bs] = acc - 0.9 * fmin(acc, 0.f);
        }