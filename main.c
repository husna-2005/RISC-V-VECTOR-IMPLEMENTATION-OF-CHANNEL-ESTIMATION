#include <printf.h>
#include <riscv_vector.h>
#include <stdint.h>
#include <math.h>


void fmatmul_16x16(double *c, const double *a, const double *b,
                   unsigned long int m, unsigned long int n,
                   unsigned long int p);
void fmatmul_vec_16x16_slice_init();
void fmatmul_vec_16x16(double *c, const double *a, const double *b,
                       unsigned long int n, unsigned long int p);


#define MIN(a, b) ((a) < (b) ? (a) : (b))

/*void generate_random_matrix(double *matrix, int rows, int cols, double max_val) {
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            matrix[i * cols + j] = ((double)rand() / RAND_MAX) * max_val;
        }
    }
}*/

void fmatmul_16x16(double *c, const double *a, const double *b,
                   unsigned long int M, unsigned long int N,
                   unsigned long int P) {
  // We work on 4 rows of the matrix at once
  const unsigned long int block_size = 16;
  unsigned long int block_size_p;

  // Set the vector configuration
  asm volatile("vsetvli %0, %1, e64, m1, ta, ma" : "=r"(block_size_p) : "r"(P));

  // Slice the matrix into a manageable number of columns p_
  for (unsigned long int p = 0; p < P; p += block_size_p) {
    // Set the vector length
    const unsigned long int p_ = MIN(P - p, block_size_p);

    // Find pointers to the submatrices
    const double *b_ = b + p;
    double *c_ = c + p;

    asm volatile("vsetvli zero, %0, e64, m1, ta, ma" ::"r"(p_));

    // Iterate over the rows
    for (unsigned long int m = 0; m < M; m += block_size) {
      // Find pointer to the submatrices
      const double *a_ = a + m * N;
      double *c__ = c_ + m * P;

      fmatmul_vec_16x16_slice_init();
      fmatmul_vec_16x16(c__, a_, b_, N, P);
    }
  }
}

void fmatmul_vec_16x16_slice_init() {
  asm volatile("vmv.v.i v0,  0");
  asm volatile("vmv.v.i v1,  0");
  asm volatile("vmv.v.i v2,  0");
  asm volatile("vmv.v.i v3,  0");
  asm volatile("vmv.v.i v4,  0");
  asm volatile("vmv.v.i v5,  0");
  asm volatile("vmv.v.i v6,  0");
  asm volatile("vmv.v.i v7,  0");
  asm volatile("vmv.v.i v8,  0");
  asm volatile("vmv.v.i v9,  0");
  asm volatile("vmv.v.i v10, 0");
  asm volatile("vmv.v.i v11, 0");
  asm volatile("vmv.v.i v12, 0");
  asm volatile("vmv.v.i v13, 0");
  asm volatile("vmv.v.i v14, 0");
  asm volatile("vmv.v.i v15, 0");
}

void fmatmul_vec_16x16(double *c, const double *a, const double *b,
                       const unsigned long int N, const unsigned long int P) {
  // Temporary variables
  double t0, t1, t2, t3, t4, t5, t6, t7, t8, t9, t10, t11, t12, t13, t14, t15;

  // Original pointer
  const double *a_ = a;

  // Prefetch one row of scalar values
  t0 = *a, a += N;
  t1 = *a, a += N;
  t2 = *a, a += N;
  t3 = *a, a += N;
  t4 = *a, a += N;
  t5 = *a, a += N;
  t6 = *a, a += N;
  t7 = *a, a += N;
  t8 = *a, a += N;
  t9 = *a, a += N;
  t10 = *a, a += N;
  t11 = *a, a += N;
  t12 = *a, a += N;
  t13 = *a, a += N;
  t14 = *a, a += N;
  t15 = *a;

  // Prefetch one row of matrix B
  asm volatile("vle64.v v16, (%0);" ::"r"(b));
  b += P;

  // Compute the multiplication
  unsigned long int n = 0;

  while (n != N) {
#ifdef VCD_DUMP
    // Start dumping VCD
    if (n == 8)
      event_trigger = +1;
    // Stop dumping VCD
    if (n == 12)
      event_trigger = -1;
#endif

    // Calculate pointer to the matrix A
    a = a_ + ++n;

    asm volatile("vfmacc.vf v0, %0, v16" ::"f"(t0));
    t0 = *a, a += N;

    // Load one row of B
    asm volatile("vle64.v v17, (%0);" ::"r"(b));
    b += P;

    asm volatile("vfmacc.vf v1, %0, v16" ::"f"(t1));
    t1 = *a, a += N;
    asm volatile("vfmacc.vf v2, %0, v16" ::"f"(t2));
    t2 = *a, a += N;
    asm volatile("vfmacc.vf v3, %0, v16" ::"f"(t3));
    t3 = *a, a += N;
    asm volatile("vfmacc.vf v4, %0, v16" ::"f"(t4));
    t4 = *a, a += N;
    asm volatile("vfmacc.vf v5, %0, v16" ::"f"(t5));
    t5 = *a, a += N;
    asm volatile("vfmacc.vf v6, %0, v16" ::"f"(t6));
    t6 = *a, a += N;
    asm volatile("vfmacc.vf v7, %0, v16" ::"f"(t7));
    t7 = *a, a += N;
    asm volatile("vfmacc.vf v8, %0, v16" ::"f"(t8));
    t8 = *a, a += N;
    asm volatile("vfmacc.vf v9, %0, v16" ::"f"(t9));
    t9 = *a, a += N;
    asm volatile("vfmacc.vf v10, %0, v16" ::"f"(t10));
    t10 = *a, a += N;
    asm volatile("vfmacc.vf v11, %0, v16" ::"f"(t11));
    t11 = *a, a += N;
    asm volatile("vfmacc.vf v12, %0, v16" ::"f"(t12));
    t12 = *a, a += N;
    asm volatile("vfmacc.vf v13, %0, v16" ::"f"(t13));
    t13 = *a, a += N;
    asm volatile("vfmacc.vf v14, %0, v16" ::"f"(t14));
    t14 = *a, a += N;
    asm volatile("vfmacc.vf v15, %0, v16" ::"f"(t15));
    t15 = *a;

    a = a_ + ++n;

    if (n == N)
      break;

    asm volatile("vfmacc.vf v0, %0, v17" ::"f"(t0));
    t0 = *a, a += N;

    // Load one row of B
    asm volatile("vle64.v v16, (%0);" ::"r"(b));
    b += P;

    asm volatile("vfmacc.vf v1, %0, v17" ::"f"(t1));
    t1 = *a, a += N;
    asm volatile("vfmacc.vf v2, %0, v17" ::"f"(t2));
    t2 = *a, a += N;
    asm volatile("vfmacc.vf v3, %0, v17" ::"f"(t3));
    t3 = *a, a += N;
    asm volatile("vfmacc.vf v4, %0, v17" ::"f"(t4));
    t4 = *a, a += N;
    asm volatile("vfmacc.vf v5, %0, v17" ::"f"(t5));
    t5 = *a, a += N;
    asm volatile("vfmacc.vf v6, %0, v17" ::"f"(t6));
    t6 = *a, a += N;
    asm volatile("vfmacc.vf v7, %0, v17" ::"f"(t7));
    t7 = *a, a += N;
    asm volatile("vfmacc.vf v8, %0, v17" ::"f"(t8));
    t8 = *a, a += N;
    asm volatile("vfmacc.vf v9, %0, v17" ::"f"(t9));
    t9 = *a, a += N;
    asm volatile("vfmacc.vf v10, %0, v17" ::"f"(t10));
    t10 = *a, a += N;
    asm volatile("vfmacc.vf v11, %0, v17" ::"f"(t11));
    t11 = *a, a += N;
    asm volatile("vfmacc.vf v12, %0, v17" ::"f"(t12));
    t12 = *a, a += N;
    asm volatile("vfmacc.vf v13, %0, v17" ::"f"(t13));
    t13 = *a, a += N;
    asm volatile("vfmacc.vf v14, %0, v17" ::"f"(t14));
    t14 = *a, a += N;
    asm volatile("vfmacc.vf v15, %0, v17" ::"f"(t15));
    t15 = *a;
  }

  // Last iteration: store results
  asm volatile("vfmacc.vf v0, %0, v17" ::"f"(t0));
  asm volatile("vse64.v v0, (%0);" ::"r"(c));
  c += P;
  asm volatile("vfmacc.vf v1, %0, v17" ::"f"(t1));
  asm volatile("vse64.v v1, (%0);" ::"r"(c));
  c += P;
  asm volatile("vfmacc.vf v2, %0, v17" ::"f"(t2));
  asm volatile("vse64.v v2, (%0);" ::"r"(c));
  c += P;
  asm volatile("vfmacc.vf v3, %0, v17" ::"f"(t3));
  asm volatile("vse64.v v3, (%0);" ::"r"(c));
  c += P;
  asm volatile("vfmacc.vf v4, %0, v17" ::"f"(t4));
  asm volatile("vse64.v v4, (%0);" ::"r"(c));
  c += P;
  asm volatile("vfmacc.vf v5, %0, v17" ::"f"(t5));
  asm volatile("vse64.v v5, (%0);" ::"r"(c));
  c += P;
  asm volatile("vfmacc.vf v6, %0, v17" ::"f"(t6));
  asm volatile("vse64.v v6, (%0);" ::"r"(c));
  c += P;
  asm volatile("vfmacc.vf v7, %0, v17" ::"f"(t7));
  asm volatile("vse64.v v7, (%0);" ::"r"(c));
  c += P;
  asm volatile("vfmacc.vf v8, %0, v17" ::"f"(t8));
  asm volatile("vse64.v v8, (%0);" ::"r"(c));
  c += P;
  asm volatile("vfmacc.vf v9, %0, v17" ::"f"(t9));
  asm volatile("vse64.v v9, (%0);" ::"r"(c));
  c += P;
  asm volatile("vfmacc.vf v10, %0, v17" ::"f"(t10));
  asm volatile("vse64.v v10, (%0);" ::"r"(c));
  c += P;
  asm volatile("vfmacc.vf v11, %0, v17" ::"f"(t11));
  asm volatile("vse64.v v11, (%0);" ::"r"(c));
  c += P;
  asm volatile("vfmacc.vf v12, %0, v17" ::"f"(t12));
  asm volatile("vse64.v v12, (%0);" ::"r"(c));
  c += P;
  asm volatile("vfmacc.vf v13, %0, v17" ::"f"(t13));
  asm volatile("vse64.v v13, (%0);" ::"r"(c));
  c += P;
  asm volatile("vfmacc.vf v14, %0, v17" ::"f"(t14));
  asm volatile("vse64.v v14, (%0);" ::"r"(c));
  c += P;
  asm volatile("vfmacc.vf v15, %0, v17" ::"f"(t15));
  asm volatile("vse64.v v15, (%0);" ::"r"(c));
}

void printnumb(double *numb, int rows, int cols) {
  for (int i = 0; i < rows; i++) {
    for (int j = 0; j < cols; j++) {
      printf("%6.1f ", numb[i * cols + j]);
    }
    printf("\n");
  }
}

int main(){

	double x[256] = {
48, 27, 31, 17, 46, 13, 28, 4, 49, 25, 41, 26, 32, 27, 13, 33,
18, 43, 42, 2, 47, 18, 42, 28, 17, 1, 5, 16, 28, 2, 25, 23,
27, 42, 27, 16, 40, 30, 14, 7, 7, 11, 36, 27, 43, 13, 31, 14,
22, 38, 22, 16, 30, 22, 38, 39, 30, 20, 45, 5, 12, 48, 40, 19,
4, 15, 31, 10, 48, 1, 14, 15, 30, 1, 2, 18, 16, 9, 31, 46,
18, 47, 47, 13, 1, 13, 10, 37, 41, 21, 12, 3, 26, 27, 29, 39,
34, 13, 40, 5, 20, 49, 23, 15, 23, 3, 45, 15, 11, 4, 9, 43,
2, 30, 5, 5, 18, 18, 1, 40, 17, 24, 42, 17, 25, 42, 16, 49,
41, 4, 7, 6, 10, 48, 6, 32, 16, 21, 6, 45, 39, 35, 3, 42,
48, 12, 13, 46, 21, 31, 2, 21, 43, 28, 36, 45, 17, 1, 24, 32,
38, 33, 43, 44, 46, 15, 22, 21, 3, 47, 3, 6, 12, 5, 15, 6,
37, 6, 27, 8, 48, 16, 32, 24, 37, 21, 28, 10, 32, 27, 17, 46,
36, 40, 39, 11, 34, 23, 18, 3, 1, 28, 38, 30, 29, 6, 28, 14,
9, 36, 48, 19, 7, 28, 15, 38, 1, 45, 38, 10, 28, 48, 33, 2,
43, 28, 26, 30, 5, 49, 17, 31, 30, 19, 31, 35, 4, 8, 16, 46,
4, 3, 3, 6, 20, 10, 8, 10, 16, 20, 8, 32, 18, 30, 42, 6 };

	double y[256] = { 
48, 27, 31, 17, 46, 13, 28, 4, 49, 25, 41, 26, 32, 27, 13, 33,
18, 43, 42, 2, 47, 18, 42, 28, 17, 1, 5, 16, 28, 2, 25, 23,
27, 42, 27, 16, 40, 30, 14, 7, 7, 11, 36, 27, 43, 13, 31, 14,
22, 38, 22, 16, 30, 22, 38, 39, 30, 20, 45, 5, 12, 48, 40, 19,
4, 15, 31, 10, 48, 1, 14, 15, 30, 1, 2, 18, 16, 9, 31, 46,
18, 47, 47, 13, 1, 13, 10, 37, 41, 21, 12, 3, 26, 27, 29, 39,
34, 13, 40, 5, 20, 49, 23, 15, 23, 3, 45, 15, 11, 4, 9, 43,
2, 30, 5, 5, 18, 18, 1, 40, 17, 24, 42, 17, 25, 42, 16, 49,
41, 4, 7, 6, 10, 48, 6, 32, 16, 21, 6, 45, 39, 35, 3, 42,
48, 12, 13, 46, 21, 31, 2, 21, 43, 28, 36, 45, 17, 1, 24, 32,
38, 33, 43, 44, 46, 15, 22, 21, 3, 47, 3, 6, 12, 5, 15, 6,
37, 6, 27, 8, 48, 16, 32, 24, 37, 21, 28, 10, 32, 27, 17, 46,
36, 40, 39, 11, 34, 23, 18, 3, 1, 28, 38, 30, 29, 6, 28, 14,
9, 36, 48, 19, 7, 28, 15, 38, 1, 45, 38, 10, 28, 48, 33, 2,
43, 28, 26, 30, 5, 49, 17, 31, 30, 19, 31, 35, 4, 8, 16, 46,
4, 3, 3, 6, 20, 10, 8, 10, 16, 20, 8, 32, 18, 30, 42, 6 };


	double zv[256] = {0};


	fmatmul_16x16(zv, x, y, 16, 16, 16);
       
	printf("\n");
        //printnumb(zv, 64);
	printf("success!\n");
        printnumb(zv, 16, 16);

  
	return 0;
}

