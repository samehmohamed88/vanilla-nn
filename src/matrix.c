#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <math.h>
#include <time.h>

#include "matrix.h"

void zero_out_data(Matrix *res, int depth, int rows, int cols)
{

  if (res->rows != rows || res->cols != cols)
    {
      if ((res->rows*res->cols) != (rows*cols))
  	{

  	  free(res->data);
  	  res->data = NULL;
	  
  	  make_matrix(res, depth, rows, cols);
  	}

      res->rows = rows;
      res->cols = cols;
      res->depth = depth;
      res->length = depth*rows*cols;

    }
  else {

    res->rows = rows;
    res->cols = cols;
    res->depth = depth;
    res->length = depth*rows*cols;
    
    for (int i =0; i < (rows*cols); i++)
      {
    	res->data[i] = 0.0;
      }
  }
}
void sigmoid(Matrix *z, Matrix *res)
{

  zero_out_data(res, z->depth, z->rows, z->cols);
  
  for (int i = 0;
       i < z->length;
       i++)
    {
      res->data[i] = 1.0/(1.0+exp(-z->data[i]));
    }

}

void matrix_add(Matrix *w, Matrix *b, Matrix *res)
{
  assert(w->rows == b->rows);
  assert(w->cols == b->cols);
  
  zero_out_data(res, w->depth, w->rows, w->cols);
  
  for (int i = 0;
       i < w->length;
       i++)
    {
      res->data[i] = w->data[i] + b->data[i];
    }
}

void matrix_sadd(Matrix *a, Matrix *b)
{
  for (int i = 0;
       i < a->length;
       i++)
    {
      a->data[i] += b->data[i];
    }
}

void matrix_mul(Matrix *a, Matrix *b, Matrix *res)
{
  assert(a->cols == b->cols);
  assert(a->rows == b->rows);

  zero_out_data(res, a->depth, a->rows, a->cols);
  
  for (int i = 0;
       i < a->length;
       i++)
    {
      res->data[i] = a->data[i]*b->data[i];
    }

}

void matrix_sub(Matrix *a, Matrix *b, Matrix *res)
{
  assert(a->cols == b->cols);
  assert(a->rows == b->rows);
  
  zero_out_data(res, a->depth, a->rows, a->cols);
  
  for (int i = 0;
       i < a->length;
       i++)
    {
      res->data[i] = a->data[i] - b->data[i];
    }
}

void sigmoid_prime(Matrix *z, Matrix *res)
{
  float one = 1.0;
  Matrix *s, *ones, *sub;

  s = malloc(sizeof(Matrix));
  ones = malloc(sizeof(Matrix));
  sub = malloc(sizeof(Matrix));
  
  make_matrix(s, z->depth, z->rows, z->cols);
  make_matrix(ones, z->depth, z->rows, z->cols);
  make_matrix(sub, z->depth, z->rows, z->cols);
  
  //memset(ones->data, one, ones->length);

  for (int i =0;
       i < ones->length;
       i++)
    {
      ones->data[i] = one;
    }
  
  sigmoid(z, s);
  
  matrix_sub(ones, s, sub);
  
  matrix_mul(s, sub, res);
 
}

void cost_derivative(Matrix *output_activations, Matrix *y, Matrix *res)
{
  matrix_sub(output_activations, y, res);
}

// Make empty matrix filled with zeros
// int rows: number of rows in matrix
// int cols: number of columns in matrix
// returns: matrix of specified size, filled with zeros
void make_matrix(Matrix *m, int depth, int rows, int cols)
{
  m->depth = depth;
  m->rows = rows;
  m->cols = cols;
  m->shallow = 0;
  m->length = depth*rows*cols;
  
  m->data = malloc(m->length * sizeof(*m->data));
 
}

// Make a matrix with uniformly random elements
// int rows, cols: size of matrix
// float s: range of randomness, [-s, s]
// returns: matrix of rows x cols with elements in range [-s,s]
void random_matrix(Matrix *m, int depth, int rows, int cols, float s, int use_s)
{

  make_matrix(m, depth, rows, cols);
  
  for (int i=0;
       i < m->length;
       i++)
    {
      if (use_s == 1)
	m->data[i] = (2*s*(rand()%1000/1000.0) - s)/sqrt(cols);
      else
	m->data[i] = 2*s*(rand()%1000/1000.0) - s;
    }

}

// Free memory associated with matrix
// matrix m: matrix to be freed
void free_matrix(Matrix *m)
{
    free(m->data);
    m->data = NULL;
    
    free(m);
    m = NULL;
}

// Copy a matrix
// matrix m: matrix to be copied
// returns: matrix that is a deep copy of m
void copy_matrix(Matrix *m, Matrix *c)
{
  
  for (int i=0;
       i < c->length;
       i++)
    {
      c->data[i] = m->data[i];
    }
}

float matrix_get(Matrix *m, int row_index, int col_index)
{
  return m->data[((m->cols*row_index)+col_index)];
}

// Transpose a matrix
// matrix m: matrix to be transposed
// returns: matrix, result of transposition
void transpose_matrix(Matrix *m, Matrix *res)
{
  zero_out_data(res, m->depth, m->cols, m->rows);
  
  for (int j=0; j < m->cols; j++)
    {
      for (int i=0; i < m->rows; i++)
	{
	  res->data[((j*m->rows)+i)] = matrix_get(m, i, j);
	}
    }
}

void matrix_set(Matrix *m, int row_index, int col_index, float val)
{
  m->data[((m->cols*row_index)+col_index)] = val;
}

void matrix_dot(Matrix *w, Matrix *x, Matrix *res)
{
  assert(w->cols == x->rows);

  zero_out_data(res, w->depth, w->rows, x->cols);
 
  int m,n,p;
  float new, cur;
  
  m = w->rows;
  n = x->cols;
  p = w->cols;
    
  for (int i=0; i < m; i++)
    {
      for (int j=0; j < n; j++)
	{
	  matrix_set(res,i,j, 0.0);
	  for (int k=0; k < p; k++)
	    {
	      cur = matrix_get(res,i,j);
	      new = matrix_get(w,i,k)*matrix_get(x,k,j);
	      matrix_set(res,i,j,(cur+new));
	    }
	}
    }
}

// Print a matrix
void print_matrix(Matrix m)
{
    int i, j;
    printf(" __");
    for(j = 0; j < 16*m.cols-1; ++j) printf(" ");
    printf("__ \n");

    printf("|  ");
    for(j = 0; j < 16*m.cols-1; ++j) printf(" ");
    printf("  |\n");

    for(i = 0; i < m.rows; ++i){
        printf("|  ");
        for(j = 0; j < m.cols; ++j){
            printf("%15.7f ", m.data[i*m.cols + j]);
        }
        printf(" |\n");
    }
    printf("|__");
    for(j = 0; j < 16*m.cols-1; ++j) printf(" ");
    printf("__|\n");
}
