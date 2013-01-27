#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <float.h>
#include <math.h>
#include <time.h>
#include <sys/time.h>
#include <cblas.h>

/* Your function must have the following signature: */

int sgemm( int m, int n, float *A, float *C );

/* The benchmarking program */


/* Copying the body of the for loop to build own test function.*/
int owntest( int a, int b )
{
    /* Allocate and fill 2 random matrices A, C */
    float *A = (float*) malloc( a * b * sizeof(float) );
    float *C = (float*) malloc( a * a * sizeof(float) );

    for( int i = 0; i < a*b; i++ ) A[i] = 2 * drand48() - 1;
    for( int i = 0; i < a*a; i++ ) C[i] = 2 * drand48() - 1;

    /* measure Gflop/s rate; time a sufficiently long sequence of calls to eliminate noise */
    double Gflop_s, seconds = -1.0;
    for( int n_iterations = 1; seconds < 0.1; n_iterations *= 2 )
    {
      /* warm-up */
      sgemm( a, b, A, C );

      /* measure time */
      struct timeval start, end;
      gettimeofday( &start, NULL );
      for( int i = 0; i < n_iterations; i++ )
	sgemm( a,b, A, C );
      gettimeofday( &end, NULL );
      seconds = (end.tv_sec - start.tv_sec) + 1.0e-6 * (end.tv_usec - start.tv_usec);

      /* compute Gflop/s rate */
      Gflop_s = 2e-9 * n_iterations * a * a * b / seconds;
    }

    printf( "%d by %d matrix \t %g Gflop/s\n", a, b, Gflop_s );

    /* Ensure that error does not exceed the theoretical error bound */

    /* Set initial C to 0 and do matrix multiply of A*B */
    memset( C, 0, sizeof( float ) * a * a );
    sgemm( a,b, A, C );

    /* Subtract A*B from C using standard sgemm (note that this should be 0 to within machine roundoff) */
    cblas_sgemm( CblasColMajor,CblasNoTrans,CblasTrans, a,a,b, -1, A,a, A,a, 1, C,a );

    /* Subtract the maximum allowed roundoff from each element of C */
    for( int i = 0; i < a*b; i++ ) A[i] = fabs( A[i] );
    for( int i = 0; i < a*a; i++ ) C[i] = fabs( C[i] );
    cblas_sgemm( CblasColMajor,CblasNoTrans,CblasTrans, a,a,b, -3.0*FLT_EPSILON*b, A,a, A,a, 1, C,a );

    /* After this test if any element in C is still positive something went wrong in square_sgemm */
    for( int i = 0; i < a * a; i++ )
      if( C[i] > 0 ) {
	printf( "FAILURE: error in matrix multiply exceeds an acceptable margin\n" );
	return -1;
      }

    /* release memory */
    free( C );
    free( A );

    return 0;
}


int main( int argc, char **argv )
{
  srand(time(NULL));

  int n = 32;
  int v = 100;
  static int nprime[7] = { 37, 43, 53, 61, 71, 79, 89 };
  static int mprime[27] = { 1597, 1613, 1637,
		  	  	  	  	  	2221, 2251, 2281,
		  	  	  	  	    3187, 3217, 3253,
		  	  	  	  	    4507, 4523, 4567,
		  	  	  	  	    5641, 5657, 5689,
		  	  	  	  	    6311, 6337, 6361,
		  	  	  	  	    7001, 7039, 7079,
		  	  	  	  	    8513, 8539, 8581,
		  	  	  	  	    9203, 9239, 9281 };
  int powOf2m = 1024, powOf2n = 32;
  int x, y;

  printf( "Own tests.\n");

//  if ( owntest ( 99 , 99 ) == -1)
//	  return -1;

  printf( "Edge case of n\n");
  /* Test [100,99] matrix. */
  if ( owntest ( 100 , 99 ) == -1)
	  return -1;
  /* Test [100,98] matrix. */
  if ( owntest ( 100 , 98 ) == -1)
	  return -1;
  /* Test [100,97] matrix. */
  if ( owntest ( 100 , 97 ) == -1)
	  return -1;
  printf( "------------------\n");

  printf( "Edge case of m\n");
  /* Test [99,100] matrix. */
  if ( owntest ( 99 , 100 ) == -1)
	  return -1;
  /* Test [98,100] matrix. */
  if ( owntest ( 98 , 100 ) == -1)
	  return -1;
  /* Test [97,100] matrix. */
  if ( owntest ( 97 , 100 ) == -1)
	  return -1;
  printf( "------------------\n");


  printf( "Power of 2 tests.\n");
  for ( x = 0; x < 4; x++ )
  {
	  powOf2n = 32;
	  for ( y = 0; y < 2; y++ )
	  {
		  if ( owntest ( powOf2m, powOf2n ) == -1)
			  return -1;
  	  	  powOf2n *= 2;
	  }
	  powOf2m *= 2;
  }
  printf( "------------------\n");

  printf( "Prime dimension tests.\n");
  for ( x = 0; x < 7; x++ )
	  for ( y = 0; y < 27; y++ )
		  if ( owntest ( mprime[y], nprime[x] ) == -1)
			  return -1;
  printf( "------------------\n");


  printf( "Bound case tests.\n");
  /* Test [1000,32] matrix. */
  if ( owntest ( 1000, 32 ) == -1)
	  return -1;
  /* Test [1000,100] matrix. */
  if ( owntest ( 1000, 100 ) == -1)
	  return -1;
  /* Test [10000,100] matrix. */
  if ( owntest ( 10000, 100 ) == -1)
	  return -1;
  /* Test [10000,32] matrix. */
  if ( owntest ( 10000, 32 ) == -1)
	  return -1;
  printf( "------------------\n");


  printf( "60 * 60 tests.\n");
  for ( x = 0; x < 20; x++)
  {
	  /* Test [60, 60] matrix. */
	  if ( owntest ( 60, 60 ) == -1)
		  return -1;
  }
  printf( "------------------\n");

  printf( "Default tests.\n");
  printf( "------------------\n");
  /* Try different m */
  for( int m = 1000; m < 10000; m = m+1+m/3 )
  {
    /* Allocate and fill 2 random matrices A, C */
    float *A = (float*) malloc( m * v * sizeof(float) );
    float *C = (float*) malloc( m * m * sizeof(float) );
    
    for( int i = 0; i < m*v; i++ ) A[i] = 2 * drand48() - 1;
    for( int i = 0; i < m*m; i++ ) C[i] = 2 * drand48() - 1;
    
    /* measure Gflop/s rate; time a sufficiently long sequence of calls to eliminate noise */
    double Gflop_s, seconds = -1.0;
    for( int n_iterations = 1; seconds < 0.1; n_iterations *= 2 ) 
    {
      /* warm-up */
      sgemm( m, v, A, C );
      
      /* measure time */
      struct timeval start, end;
      gettimeofday( &start, NULL );
      for( int i = 0; i < n_iterations; i++ )
	sgemm( m,v, A, C );
      gettimeofday( &end, NULL );
      seconds = (end.tv_sec - start.tv_sec) + 1.0e-6 * (end.tv_usec - start.tv_usec);
      
      /* compute Gflop/s rate */
      Gflop_s = 2e-9 * n_iterations * m * m * v / seconds;
    }
    
    printf( "%d by %d matrix \t %g Gflop/s\n", m, v, Gflop_s );
    
    /* Ensure that error does not exceed the theoretical error bound */
		
    /* Set initial C to 0 and do matrix multiply of A*B */
    memset( C, 0, sizeof( float ) * m * m );
    sgemm( m,v, A, C );

    /* Subtract A*B from C using standard sgemm (note that this should be 0 to within machine roundoff) */
    cblas_sgemm( CblasColMajor,CblasNoTrans,CblasTrans, m,m,v, -1, A,m, A,m, 1, C,m );

    /* Subtract the maximum allowed roundoff from each element of C */
    for( int i = 0; i < m*v; i++ ) A[i] = fabs( A[i] );
    for( int i = 0; i < m*m; i++ ) C[i] = fabs( C[i] );
    cblas_sgemm( CblasColMajor,CblasNoTrans,CblasTrans, m,m,v, -3.0*FLT_EPSILON*v, A,m, A,m, 1, C,m );

    /* After this test if any element in C is still positive something went wrong in square_sgemm */
    for( int i = 0; i < m * m; i++ )
      if( C[i] > 0 ) {
	printf( "FAILURE: error in matrix multiply exceeds an acceptable margin\n" );
	return -1;
      }

    /* release memory */
    free( C );
    free( A );
  }
  printf( "------------------\n");

  return 0;
}
