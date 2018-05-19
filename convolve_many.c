#define FILTER_EDGE_SIZE 3

__kernel void convolve_many(
            const unsigned int bx, // block x size
            const unsigned int by, // block y size
            const unsigned int num_inputs, // number of input pixels
            const unsigned int num_outputs,// number of output pixels
            global const float *in, // input buffer
            global const float *k, // input filter
            global const float *bias, // input bias matrix
            global float *out) // output buffer
        {
            int bs = bx*by; // block size
            int xy = get_global_id(0) + get_global_id(1) * bx;
            int z = get_global_id(2); // corresponding output pos

            float sum = 0;
            k += z * num_inputs * FILTER_EDGE_SIZE * FILTER_EDGE_SIZE;
            int i_off = xy; //The top-left most position of the pixels which this compute-item need to compute
            for (int i = 0; i < num_inputs; i++) {
                for (int k_i = 0;k_i < FILTER_EDGE_SIZE * FILTER_EDGE_SIZE;k_i++) {
                    sum += in[i_off + (k_i / FILTER_EDGE_SIZE) * bx + k_i % FILTER_EDGE_SIZE] * (*(k++));
                }
                i_off += bs; // move to next block
            }
            sum += bias[z];
            out[xy + z*bs] = sum - 0.9 * fmin(sum, 0.f); // apply the Rectified Linear Unit
        }