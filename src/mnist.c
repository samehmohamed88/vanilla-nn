#define _POSIX_C_SOURCE 1

#include <stdio.h>
#include <stdlib.h>

#include <unistd.h>
#include <sysexits.h>
#include <fcntl.h>
#include <errno.h>
#include <string.h>

#include <sys/types.h>
#include <sys/stat.h>
#include <math.h>
#include <time.h>

#include <arpa/inet.h>
#define betoh32 ntohl

#include "matrix.h"
#include "mnist.h"

void write_image(Matrix *image, Matrix *label, int x)
{
  FILE *fp;
  u_char w;
  char pname[70];
  int yactual;

  for (int j = 0; j < (label->rows*label->cols); j++)
    {
      if (label->data[j] == 1.0)
	{
	  yactual = j;
	}
    }


  snprintf(pname, 72, "%d-%d.pgm", yactual, x);
  fp = fopen(pname, "w");
  fprintf(fp, "P5\n");
  fprintf(fp, "# %s\n", pname);
  fprintf(fp, "%u %u\n", 28, 28);
  fprintf(fp, "255\n");
  fflush(fp);
      
  for (int i = 0; i < 28; i++) {
    for (int j = 0; j < 28; j++) {
      w = image->data[j + (i * 28)];
      write(fileno(fp), &w, 1);
    }
    fprintf(fp, "\n");
  }
  fprintf(fp, "\n");
  fclose(fp);
}

void shuffle_data(Matrix **images, Matrix **labels, int count)
{
  int x1,x2;
  Matrix *img, *lbl;

  for (int z = 0; z < count; z++)
    {      
      x1 = rand()%count;
      x2 = rand()%count;

      img = images[x2];
      lbl = labels[x2];

      images[x2] = images[x1];
      labels[x2] = labels[x1];

      images[x1] = img;
      labels[x1] = lbl;
    }
}


void get_next_label(Matrix *m, int f)
{
  int c;
  u_char byte;

  c = read(f, &byte, sizeof(byte));
  if (c < 0) {
    fprintf(stderr, "read image %s\n", strerror(errno));
    exit(EX_IOERR);
  }

  m->data[byte] = 1.0;
  
}

void
read_next_image(Matrix *m, int f, u_char *image, int ilen)
{
  int c;
  
  c = read(f, image, ilen);

  if (c < 0) {
    fprintf(stderr, "read image %s\n", strerror(errno));
    exit(EX_IOERR);
  }
  
  for (int i = 0; i < ilen; i++)
    {
      m->data[i] = (float)(*(image+i));
    }
}

void load_mnist(Data *data, char *ifile, char *lfile)
{
  
  int c, fl, fi;

  u_int32_t lmagic, imagic, lnum, inum, nr, nc, ilen, n;
  u_char *image;

  fl = open(lfile, O_RDONLY, 0);
  fi = open(ifile, O_RDONLY, 0);
  
  /* get the magic numbers */
  c = read(fl, &lmagic, sizeof(lmagic));
  lmagic = betoh32(lmagic); /* idx files are big endian */
  c = read(fi, &imagic, sizeof(imagic));
  imagic = betoh32(imagic); /* idx files are big endian */

  /* read the number of items */
  c = read(fl, &lnum, sizeof(lnum));
  lnum = betoh32(lnum);
  c = read(fi, &inum, sizeof(inum));
  inum = betoh32(inum);
  
  /* At the very least lnum and inum are equal */
  if (lnum != inum) {
    fprintf(stderr, "Please use label and image files that at least have an equal number of items!\n");
    exit(1);
  }

  /* read the number of rows */
  c = read(fi, &nr, sizeof(nr));
  nr = betoh32(nr);
  
  /* read the number of columns */
  c = read(fi, &nc, sizeof(nc));
  nc = betoh32(nc);

  /* allocate the image buffer */
  ilen = nr * nc;
  image = malloc(ilen);
  if (image == NULL) {
    fprintf(stderr, "malloc(image): %s\n", strerror(errno));
    exit(EX_OSERR);
  }
  
#if 0
  printf("magic numbers: %u %u\n", lmagic, imagic);
  printf("#items: %u %u\n", lnum, inum);
  printf("#rows: %u #columns: %u\n", nr, nc);
#endif

  n = 0;
  //lnum = 1000;
  
  data->images = malloc(inum * sizeof(Matrix*));
  data->labels = malloc(lnum * sizeof(Matrix*));

  while (n != lnum) {
    //while (n != 10) {

    data->labels[n] = malloc(sizeof(Matrix));
    make_matrix(data->labels[n], 1, 10, 1);
    
    data->images[n] = malloc(sizeof(Matrix));
    make_matrix(data->images[n], 1, nr*nc, 1);

    get_next_label(data->labels[n], fl);
    read_next_image(data->images[n], fi, image, nr * nc);
    n++;
  }

  data->nsamples = lnum;
  
  close(fi);
  close(fl);

}

