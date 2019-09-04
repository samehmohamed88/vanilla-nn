#include <sys/types.h>
#include <string.h>

#include "matrix.h"

typedef struct Data{
  Matrix **images;
  Matrix **labels;
  int nsamples;
} Data;


typedef struct Network{

  int num_layers;
  int *sizes;

  Matrix **weights;
  Matrix **biases;

  Data *training_data;
  Data *validation_data;
  Data *test_data;

} Network;

typedef struct BackpropArgs{
  int start;
  int end;
  Network *network;
  Matrix **image_delta_nabla_w;
  Matrix **image_delta_nabla_b;
  Matrix **delta_nabla_w;
  Matrix **delta_nabla_b;
  Matrix *activation, *m, *d, *sp, *delta;
  Matrix **activations, **zs;
} BackpropArgs;


void read_next_image(Matrix *m, int f, u_char *image, int ilen);

void load_mnist(Data *data, char *ifile, char *lfile);

void get_next_label(Matrix *m, int f);

void shuffle_data(Matrix **images, Matrix **labels, int count);

void write_image(Matrix *image, Matrix *label, int x);

void sgd(Network *network, int num_epochs, int batch_size, float learning_rate, float lambda);
