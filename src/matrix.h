// Include guards and C++ compatibility
#ifndef MATRIX_H
#define MATRIX_H
#ifdef __cplusplus
extern "C" {
#endif

// A Matrix has size rows x cols
// and some data stored as an array of floats
// storage is row-major order:
// https://en.wikipedia.org/wiki/Row-_and_column-major_order
typedef struct Matrix{
  int depth,rows, cols;
  float *data;
  int shallow;
  int length;
} Matrix;

void zero_out_data(Matrix *res, int depth, int rows, int cols);

void transpose_matrix(Matrix *m, Matrix *res);
  
float matrix_get(Matrix *m, int row_index, int col_index);

void  matrix_set(Matrix *m, int row_index, int col_index, float val);

void matrix_add(Matrix *w, Matrix *b, Matrix *res);

void matrix_sadd(Matrix *a, Matrix *b);

void matrix_dot(Matrix *w, Matrix *x, Matrix *res);

void sigmoid(Matrix *z, Matrix *res);

void sigmoid_prime(Matrix *z, Matrix *res);

void cost_derivative(Matrix *output_activations, Matrix *y, Matrix *res);

void matrix_sub(Matrix *a, Matrix *b, Matrix *res);

void matrix_mul(Matrix *a, Matrix *b, Matrix *res);
  
// Make empty matrix filled with zeros
// int rows: number of rows in matrix
// int cols: number of columns in matrix
// returns: matrix of specified size, filled with zeros
  void  make_matrix(Matrix *m, int depth, int rows, int cols);

// Make a matrix with uniformly random elements
// int rows, cols: size of matrix
// float s: range of randomness, [-s, s]
// returns: matrix of rows x cols with elements in range [-s,s]
  void random_matrix(Matrix *m, int depth, int rows, int cols, float s, int use_s);

// Free memory associated with matrix
// matrix m: matrix to be freed
void free_matrix(Matrix *m);

// Copy a matrix
// matrix m: matrix to be copied
// returns: matrix that is a deep copy of m
void copy_matrix(Matrix *m, Matrix *c);

// Print a matrix
void print_matrix(Matrix m);

#ifdef __cplusplus
}
#endif
#endif
