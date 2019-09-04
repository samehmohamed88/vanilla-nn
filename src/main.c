#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <pthread.h>

#include "matrix.h"
#include "mnist.h"

#define nthreads 4U

void evaluate(Network *network);

void NetworkInit(Network *network);

void
NetworkInit(Network *network)
{
    
  network->num_layers = 3;
  
  network->sizes = malloc(network->num_layers * sizeof(int));
  network->weights = malloc((network->num_layers-1) * sizeof(Matrix*));
  network->biases = malloc((network->num_layers-1) * sizeof(Matrix*));

  network->training_data = malloc(sizeof(Data));
  network->test_data = malloc(sizeof(Data));
  
  network->sizes[0] = 784;
  network->sizes[1] = 30;
  network->sizes[2] = 10;
   
  for (int i=0; i<(network->num_layers-1); i++)
    {
      network->weights[i] = malloc(sizeof(Matrix));
      network->biases[i] = malloc(sizeof(Matrix));
      
      random_matrix(network->weights[i], 1, network->sizes[i+1], network->sizes[i], 2., 0);
      random_matrix(network->biases[i], 1, network->sizes[i+1], 1, 2., 1);
    }

  load_mnist(network->training_data, "./train-images-idx3-ubyte", "./train-labels-idx1-ubyte");
  load_mnist(network->test_data, "./t10k-images-idx3-ubyte", "./t10k-labels-idx1-ubyte");

  /* for (int i = 0; i < 10;i++) */
  /*   { */
  /*     write_image(network->training_data->images[i], network->training_data->labels[i], 30+i); */
  /*     write_image(network->test_data->images[i], network->test_data->labels[i], 1); */
  /*   } */
}


void
update_batch_weights(
		     Network *network,
		     Matrix **batch_nabla_w,
		     Matrix **batch_nabla_b,
		     Matrix **delta_nabla_w,
		     Matrix **delta_nabla_b
		     )
{

  for (int i = 0;
       i < network->num_layers-1;
       i++)
    {
      /* for (int j = 0; */
      /* 	   j < (batch_nabla_w[i]->rows*batch_nabla_w[i]->cols); */
      /* 	   j++) */
      /* 	{ */
      /* 	  batch_nabla_w[i]->data[j] +=  delta_nabla_w[i]->data[j]; */
      /* 	} */

      /* for (int j = 0; */
      /* 	   j < (batch_nabla_b[i]->rows*batch_nabla_b[i]->cols); */
      /* 	   j++) */
      /* 	{ */
      /* 	  batch_nabla_b[i]->data[j] +=  delta_nabla_b[i]->data[j]; */
      /* 	} */

      matrix_sadd(batch_nabla_w[i], delta_nabla_w[i]);
      matrix_sadd(batch_nabla_b[i], delta_nabla_b[i]);

      
    }

    for (int i = 0;
       i < network->num_layers-1;
       i++)
      {
	zero_out_data(
		      delta_nabla_w[i],
		      delta_nabla_w[i]->depth,
		      delta_nabla_w[i]->rows,
		      delta_nabla_w[i]->cols
		      );
	zero_out_data(
		      delta_nabla_b[i],
		      delta_nabla_b[i]->depth,
		      delta_nabla_b[i]->rows,
		      delta_nabla_b[i]->cols
		      );
      }
  
}

void
backprop(
	 Network *network,
	 Matrix *image,
	 Matrix *label,
	 Matrix **nabla_w,
	 Matrix **nabla_b,
	 Matrix **activations,
	 Matrix **zs,
	 Matrix *activation,
	 Matrix *m,
	 Matrix *d,
	 Matrix *sp,
	 Matrix *delta
	 )
{
 
  /* use the image as the first layer's activation */
  zero_out_data(activation, 1, network->sizes[0], 1);
  copy_matrix(image, activation);
  copy_matrix(image, activations[0]);
  
  /* forward pass */
  for (int i=0; i < (network->num_layers-1); i++)
    {
      matrix_dot(network->weights[i], activation, d);
      matrix_add(d, network->biases[i], zs[i]);
      sigmoid(zs[i], activation);
      sigmoid(zs[i], activations[i+1]);

    }
  /* end forward pass */

  /* backward pass */
  cost_derivative(activations[network->num_layers-1], label, delta);
  copy_matrix(delta, nabla_b[network->num_layers-2]);
  transpose_matrix(activations[network->num_layers-2], d);
  matrix_dot(delta, d, nabla_w[network->num_layers-2]);

  for (int k = 2; k < network->num_layers; k++)
    {
      d = zs[network->num_layers-(k+1)];
      sigmoid_prime(d, sp);
      transpose_matrix(network->weights[network->num_layers-k], d);
      matrix_dot(d, delta, m);
      matrix_mul(m, sp, delta);
      copy_matrix(delta, nabla_b[(network->num_layers)-(k+1)]);
      transpose_matrix(activations[(network->num_layers)-(k+1)], d);
      matrix_dot(delta, d, nabla_w[(network->num_layers)-(k+1)]);      
    }
  /* end backwards pass */
}

void*
backprop_runner(void *arg)
{
  BackpropArgs *parg = (BackpropArgs*) arg;  
  
  for (int i = parg->start;
       i < parg->end;
       i++)
    {
      /* call backprop */
      backprop(
	       parg->network,
	       parg->network->training_data->images[i],
	       parg->network->training_data->labels[i],
	       parg->image_delta_nabla_w,
	       parg->image_delta_nabla_b,
	       parg->activations,
	       parg->zs,
	       parg->activation,
	       parg->m,
	       parg->d,
	       parg->sp,
	       parg->delta
	       );

      update_batch_weights(parg->network,
			   parg->delta_nabla_w,
			   parg->delta_nabla_b,
			   parg->image_delta_nabla_w,
			   parg->image_delta_nabla_b
			   );
      
    }
  return 0;
}

void
update_network_weights(
		       Network *network,
		       Matrix **batch_nabla_w,
		       Matrix **batch_nabla_b,
		       float learning_rate,
		       float lambda,
		       int batch_size
		       )
{

  float w, b, nw, nb,t;

  t = (1-((learning_rate*lambda)/batch_size));
    
  for (int i = 0;
       i < network->num_layers-1;
       i++)
    {
      for (int j = 0;
	   j < network->weights[i]->length;
	   j++)
	{
	  
	  w = t * network->weights[i]->data[j];
	  nw = batch_nabla_w[i]->data[j];
	  network->weights[i]->data[j] = w-((learning_rate/batch_size)*nw);
	}

      for (int j = 0;
	   j < network->biases[i]->length;
	   j++)
	{
	  b = network->biases[i]->data[j];
	  nb = batch_nabla_b[i]->data[j];
	  network->biases[i]->data[j] = b-((learning_rate/batch_size)*nb);
	}
    }
  
  for (int i = 0;
       i < network->num_layers-1;
       i++)
    {

      zero_out_data(
		    batch_nabla_w[i],
		    batch_nabla_w[i]->depth,
		    batch_nabla_w[i]->rows,
		    batch_nabla_w[i]->cols
		    );
      
      zero_out_data(
		    batch_nabla_b[i],
		    batch_nabla_b[i]->depth,
		    batch_nabla_b[i]->rows,
		    batch_nabla_b[i]->cols
		    );
      
    }
}

void
init_matrices(
	      Network *network,
	      Matrix **batch_nabla_w,
	      Matrix **batch_nabla_b,
	      BackpropArgs **args
	      )
{

  for (int i =0 ; i < nthreads; i++)
    {
      args[i]->activation = malloc(sizeof(Matrix));
      make_matrix(args[i]->activation, 1, network->sizes[0], 1);

      args[i]->activations[0] = malloc(sizeof(Matrix));
      make_matrix(args[i]->activations[0], 1, network->sizes[0], 1);
    }

  for (int i = 0; i < (network->num_layers-1); i++)
    {
      batch_nabla_w[i] = malloc(sizeof(Matrix));
      batch_nabla_b[i] = malloc(sizeof(Matrix));
      make_matrix(batch_nabla_w[i], 1, network->sizes[i+1], network->sizes[i]);
      make_matrix(batch_nabla_b[i], 1, network->sizes[i+1], 1);

      for (int j = 0; j < nthreads; j++)
	{	  
	  args[j]->delta_nabla_w[i] = malloc(sizeof(Matrix));
	  args[j]->delta_nabla_b[i] = malloc(sizeof(Matrix));

	  args[j]->image_delta_nabla_w[i] = malloc(sizeof(Matrix));
	  args[j]->image_delta_nabla_b[i] = malloc(sizeof(Matrix));
	  
	  make_matrix(args[j]->delta_nabla_w[i], 1, network->sizes[i+1], network->sizes[i]);
	  make_matrix(args[j]->delta_nabla_b[i], 1, network->sizes[i+1], 1);

	  make_matrix(args[j]->image_delta_nabla_w[i], 1, network->sizes[i+1], network->sizes[i]);
	  make_matrix(args[j]->image_delta_nabla_b[i], 1, network->sizes[i+1], 1);

	  args[j]->zs[i] = malloc(sizeof(Matrix));
	  make_matrix(args[j]->zs[i], 1, network->sizes[i+1], 1);
      
	  args[j]->activations[i+1] = malloc(sizeof(Matrix));
	  make_matrix(args[j]->activations[i+1], 1, network->sizes[i+1], 1);
	}
    }

  for (int i =0 ; i < nthreads; i++)
    {
      args[i]->m = malloc(sizeof(Matrix));
      args[i]->d = malloc(sizeof(Matrix));
      args[i]->sp = malloc(sizeof(Matrix));
      args[i]->delta = malloc(sizeof(Matrix));
      
      make_matrix(args[i]->m, 1, 2 ,2);
      make_matrix(args[i]->d, 1, 2, 2);
      make_matrix(args[i]->sp, 1, 2, 2);
      make_matrix(args[i]->delta, 1, 2, 2);
    }
}

void sgd(
	 Network *network,
	 int num_epochs,
	 int batch_size,
	 float learning_rate,
	 float lambda
	 )
{

  int last, avg;

  BackpropArgs **args;
  Matrix **batch_nabla_w, **batch_nabla_b;

  args = malloc(nthreads * sizeof(*args));
  
  batch_nabla_w = malloc((network->num_layers-1)*sizeof(Matrix*));
  batch_nabla_b = malloc((network->num_layers-1)*sizeof(Matrix*));

  for (int i =0 ; i < nthreads; i++)
    {
      args[i] = malloc(sizeof(*args[i]));

      args[i]->network = network;
      args[i]->image_delta_nabla_w = malloc((network->num_layers-1)*sizeof(Matrix*));
      args[i]->image_delta_nabla_b = malloc((network->num_layers-1)*sizeof(Matrix*));

      args[i]->delta_nabla_w = malloc((network->num_layers-1)*sizeof(Matrix*));
      args[i]->delta_nabla_b = malloc((network->num_layers-1)*sizeof(Matrix*));

      args[i]->activations = malloc((network->num_layers)*sizeof(Matrix*));
      args[i]->zs = malloc((network->num_layers-1)*sizeof(Matrix*));
    }
  
  init_matrices(
		network,
		batch_nabla_w,
		batch_nabla_b,
		args
		);

      
  pthread_attr_t attr;
  pthread_attr_init(&attr);

  avg = batch_size / nthreads;  

  for (int i = 0; i < num_epochs; i++)
    {
      printf(">> training epoch %d\n", i+1);
   
      for (int j = 0;
	   j < network->training_data->nsamples;
	   j += batch_size)
	{
	  
	  last = 0;
      
	  pthread_t tids[nthreads];
      
	  for (int k = 0; k < nthreads; k++) {
	    args[k]->start = j + last;
	    
	    if (k == 3)
	      args[k]->end = j + batch_size;
	    else
	      args[k]->end = args[k]->start + avg;
	
	    last += avg;
	
	    pthread_create(&tids[k], &attr, backprop_runner, args[k]);
	  }

	  // Wait until threads have finished their work
	  for (int k = 0; k < nthreads; k++) {
	    pthread_join(tids[k], NULL);
	  }

	  // combine the deltas from all threads
	  for (int k = 0; k < nthreads; k++) {
	    update_batch_weights(
				 network,
				 batch_nabla_w,
				 batch_nabla_b,
				 args[k]->delta_nabla_w,
				 args[k]->delta_nabla_b);
	  }

	  // update network weights (its end of batch)
	  update_network_weights(
				 network,
				 batch_nabla_w,
				 batch_nabla_b,
				 learning_rate,
				 lambda,
				 batch_size
				 );      
	}
      //evaluate(network);
    }
  evaluate(network);
}

int argmax(Matrix* y)
{
  int ynum;
    //memset(res->data, 0.0, res->length);
  for (int i =0; i < y->rows*y->cols; i++)
    {
      if (y->data[i] == 1.0)
	{
	  ynum = i;
	}
    }
  return ynum;
}

void evaluate(Network *network)
{
  float ypred;
  int correct = 0;
  int ynum;
  int yactual;

  Matrix *add, *dot, *a;

  add = malloc(sizeof(Matrix));
  make_matrix(add, 1, 2, 2);
  
  dot = malloc(sizeof(Matrix));
  make_matrix(dot, 1, 2, 2);

  a = malloc(sizeof(Matrix));
  make_matrix(a, 1, network->sizes[0], 1);

  for (int i =0; i < network->test_data->nsamples; i++)
  //for (int i =0; i < 10; i++)
    {

      zero_out_data(a, 1, network->sizes[0], 1);
      copy_matrix(network->test_data->images[i], a);
     
      for (int j=0; j < network->num_layers-1; j++)
	{
	  matrix_dot(network->weights[j], a, dot);
	  matrix_add(dot, network->biases[j], add);
	  sigmoid(add, a);
	}

      ypred = a->data[0];
      ynum = 0;
      for (int j=1; j < a->rows*a->cols; j++)
	{
	  if (a->data[j] > ypred)
	    {
	      ynum = j;
	      ypred = a->data[j];
	    }
	}     
      
      for (int j = 0;
	   j < (network->test_data->labels[i]->length);
	   j++)
	{
	  if (network->test_data->labels[i]->data[j] == 1.0)
	    {
	      yactual = j;
	    }
	}

      if (ynum == yactual)
	correct++;
      
    }
  
  printf("accuracy %f\n", (100*(float)correct/(float)network->test_data->nsamples));

}

int main()
{
  srand ( time(NULL) );

  Network *network;

  network = malloc(sizeof(Network));
  
  NetworkInit(network);
  
  sgd(network, 30, 100, .1, .1);
  
}
