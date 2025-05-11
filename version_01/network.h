#include <stdio.h>
#include <stdlib.h>
#include <time.h>

typedef struct neuron_struct {
    int num_weights;
    float *weights;
    float bias;
    float z;
    float a;
} neuron;

typedef struct layer_struct {
    int num_neurons;
    neuron *neurons;
} layer;

typedef struct network_struct {
    int num_layers;
    layer *layers;
} network;

typedef struct weight_indices_struct {
    int **w_indices;
} weight_indices;

/* TODO: create structs instead of the quadruple pointer thing for random indices */
typedef struct random_indices_struct {
    int **n_indices;
    weight_indices *w_indices;
} random_indices;

void free_network_memory(network *net, int ****rand_indices);

neuron create_neuron(int num_out_weights);

layer create_layer(int num_neurons);

network create_network(int num_layers);

network construct_network(int num_outputs, int num_layers, int *num_neurons);

void forward_layer(network net, int layer_idx);

void forward(network net, int dummy_operations);

void forward_shuffled(network net, int dummy_operations);

// No Overhead
void forward_shuffled_NO(network net, int**** random_indices, int dummy_operations);

// No Overhead, Activations At the End, Random Dummy Operations
void forward_shuffled_NO_AAE(network net, int**** random_indices, int dummy_operations);

// No Overhead, Activations At the End, Random Dummy Operations
void forward_shuffled_NO_AAE_RDO(network net, int**** random_indices, int ***random_dummy_operations);


//RANDOM SHUFFLING
void swap(int *a, int *b);

void shuffleArray(int arr[], int size);

int ****generate_random_indices(network net);

int* get_random_indices(int size);

//RANDOM DUMMY OPERATIONS

int ***generate_random_dummy_operations(network net);

int* get_random_binary_indices(int size);



