#define NET_NUM_LAYERS 4
#define NET_NUM_NEURONS ((int[]){7,5,4,3})

struct {
    float lay0_weights[7][1];
    float lay1_weights[5][7];
    float lay2_weights[4][5];
    float lay3_weights[3][4];
} net_config_weights;

void *net_config_layer_weights[NET_NUM_LAYERS];

void init_weights() {
    float lay1_weights[5][7] = 
    {
        {1.43, -0.49, 0.99, -0.21, 0.12, 0.02, -0.06},
        {-0.31, 1.66, -1.09, 0.92, 1.45, -0.67, 1.02},
        {0.75, -0.89, 1.03, -1.45, 1.12, -0.58, 1.72},
        {-1.91, 1.25, 0.46, 1.88, -0.43, -1.14, 0.99},
        {1.39, -0.57, -1.66, 0.31, 0.98, 1.01, -0.76}
    };
    float lay2_weights[4][5] =
    {
        {-1.47, 0.56, 1.85, -0.91, 0.23},
        {1.17, -1.38, 0.97, 0.63, -0.14},
        {-0.88, 1.09, -1.72, 0.21, 1.57},
        {1.86, -1.06, 0.45, -0.75, 1.02}
    };
    float lay3_weights[3][4] =
    {
        {0.45, -1.89, 1.68, 0.94},
        {-0.29, 1.23, -1.47, 0.33},
        {1.54, 0.11, -0.88, 1.77},
    };
    for (int i = 0; i < 5; i++) {
        for (int j = 0; j < 7; j++) {
            net_config_weights.lay1_weights[i][j] = lay1_weights[i][j];
        }
    }
    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 5; j++) {
            net_config_weights.lay2_weights[i][j] = lay2_weights[i][j];
        }
    }
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            net_config_weights.lay3_weights[i][j] = lay3_weights[i][j];
        }
    }
    net_config_layer_weights[0] = (void*)net_config_weights.lay0_weights;
    net_config_layer_weights[1] = (void*)net_config_weights.lay1_weights;
    net_config_layer_weights[2] = (void*)net_config_weights.lay2_weights;
    net_config_layer_weights[3] = (void*)net_config_weights.lay3_weights;
}

    // PREVIOUSLY USED WEIGHTS
    // float lay1_weights[5][7] = 
    // {
    // {0.12, -1.36, 0.78, -0.44, 1.87, 0.55, -1.98},
    // {-0.31, 1.66, -1.09, 0.92, 1.45, -0.67, 1.02},
    // {0.75, -0.89, 1.03, -1.45, 1.12, -0.58, 1.72},
    // {-1.91, 1.25, 0.46, 1.88, -0.43, -1.14, 0.99},
    // {1.39, -0.57, -1.66, 0.31, 0.98, 1.01, -0.76}
    // };
    // float lay4_weights[5][3] =  {
    // {1.24, -1.57, 0.89},
    // {-0.43, 1.91, -1.04},
    // {0.12, 1.39, -1.76},
    // {-1.32, 0.98, -0.47},
    // {1.05, -0.19, 1.72}
    // };
    // SOME OLD CODE
    // for (int i = 0; i < 7; i++) {
    //     for (int j = 0; j < 1; j++) {
    //         net_config_weights.lay0_weights[i][j] = lay0_weights[i][j];
    //     }
    // }
    // for (int i = 0; i < 5; i++) {
    //     for (int j = 0; j < 1000; j++) {
    //         net_config_weights.lay1_weights[i][j] = (float)((rand() % 401) - 200) / 100;//lay1_weights[i][j];
    //     }
    // }
    // for (int i = 0; i < 5; i++) {
    //     for (int j = 0; j < 3; j++) {
    //         net_config_weights.lay4_weights[i][j] = lay4_weights[i][j];
    //     }
    // }
    //net_config_layer_weights[4] = (void*)net_config_weights.lay4_weights;


// PREVIOUSLY USED DEFINITIONS
// #define NET_NUM_LAYERS 5
// #define NET_NUM_NEURONS ((int[]){7,5,4,3,5})
// struct {
//     float lay0_weights[7][1];
//     float lay1_weights[5][7];
//     float lay2_weights[4][5];
//     float lay3_weights[3][4];
//     float lay4_weights[5][3];
// } net_config_weights;

//#define NET_NUM_NEURONS ((int[]){1000,5,4,3})
// struct {
    // float lay0_weights[1000][1];
    // float lay1_weights[5][1000];
//     float lay2_weights[4][5];
//     float lay3_weights[3][4];
// } net_config_weights;