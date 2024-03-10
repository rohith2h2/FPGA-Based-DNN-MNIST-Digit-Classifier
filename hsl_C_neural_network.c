#include <stdio.h>
#include "/proj/cad/intelFPGA_lite/18.0/hls/include/HLS/hls.h"
#include "input_0.h"
#include "input_1.h"
#include "input_2.h"
#include "input_3.h"
#include "input_4.h"
#include "input_5.h"
#include "input_6.h"
#include "input_7.h"
#include "input_8.h"
#include "input_9.h"
#include "layer_bias.h"
#include "layer_weight.h"

hls_avalon_slave_component component;

// Function for prediction
int my_predict(
    hls_avalon_slave_memory_argument(784 * sizeof(float)) float *img,
    hls_avalon_slave_memory_argument(10 * sizeof(float)) float *b,
    hls_avalon_slave_memory_argument(784 * 10 * sizeof(float)) float *w)
{
    float res[10] = {0}; // Array to store intermediate results

    // Forward pass
    //neural network
    for (int i = 0; i < 10; i++)
    {
        for (int j = 0; j < 784; j++)
        {
            res[i] += img[j] * w[i + j * 10]; // w1x1 + w2x2 + ... + wnxn
        }
        res[i] += b[i]; // Add bias
    }

    // Find the index with the maximum value
    float max_val = res[0];
    int max_idx = 0;
    
    for (int i = 1; i < 10; i++)
    {
        if (res[i] > max_val)
        {
            max_val = res[i];
            max_idx = i;
        }
    }

    return max_idx; // Return the predicted class
}

int main()
{
    float *a[10] = {input_0, input_1, input_2, input_3, input_4, input_5, input_6, input_7, input_8, input_9};

    // Loop through the inputs and print the predictions
    for (int i = 0; i < 10; i++)
    {
        int res = my_predict(a[i], layer_bias, layer_weight); // Call the prediction function
        printf("input_%d.h Predicted Result: %d\n", i, res);
    }

    return 0;
}

