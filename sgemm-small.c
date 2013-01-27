#include <stdlib.h>
#include <emmintrin.h>

void sgemm( int m, int n, float *A, float *C )
{
	register int i, j, k;	//declare indices.
	register int u, v;	//declare padded dimensions.
	float *finalA;	//padded input matrix.
	float *finalC;	//padded output matrix.

	register int offset1, offset2; //offset for memory accessing.
	register int bound; //bound for for loop iteration.

	const int blocksize = 4;	//register block sizes.
	register __m128 C_1, C_2, C_3, C_4;	//holders for vectors in output matrix.
	register __m128 A_1, A_2, A_3, A_4;	//holders for vectors in input matrix.
	register __m128	B_1, B_2, B_3, B_4;	//holders for vectors in transposed input matrix.

	//padding dimension m to the multiple of 4.
	if ( m % 4 == 0 )
	{
		u = m;
		finalC = C;
	}
	else
	{
		u = 4+m/4*4;
		finalC = (float*) calloc (u*u, sizeof(float));
	}

	//padding dimension n to the multiple of 4.
	if ( n % 4 == 0 )
	{
		v = n;
		finalA = ( u == m ) ? A : (float*) calloc (u*v, sizeof(float));
	}
	else
	{
		v = 4+n/4*4;
		finalA = (float*) calloc (u*v, sizeof(float));
	}

	//Padding input matrix.
	if ( (u!=m) || (v!=n) )
	{
		for ( j = 0; j < n; j++ )
		{
			bound = m/10*10;
			offset1 = j*u;
			offset2 = j*m;
			for ( i = 0; i < bound; i+=10 )
			{
				finalA[offset1] = A[offset2];
				finalA[offset1+1] = A[offset2+1];
				finalA[offset1+2] = A[offset2+2];
				finalA[offset1+3] = A[offset2+3];
				finalA[offset1+4] = A[offset2+4];
				finalA[offset1+5] = A[offset2+5];
				finalA[offset1+6] = A[offset2+6];
				finalA[offset1+7] = A[offset2+7];
				finalA[offset1+8] = A[offset2+8];
				finalA[offset1+9] = A[offset2+9];
				offset1 += 10;
				offset2 += 10;
			}
			for ( i = bound; i < m; i++ )
			{
				finalA[i+j*u] = A[i+j*m];
			}
		}
	}


	/*	Multiplying matrices.
	 *	1.	loop ordering.
	 *	2.	SEE instructions.
	 *	3.	Register blocking.
	 *	4. 	loop unrolling.
	 */
	for ( j = 0; j < u; j+=blocksize )
	{

		//loop unrolling by a factor of 4.
		for ( i = 0; i < u; i+=blocksize )
		{

			offset1 = i+j*u;
			//load final matrix variables in vector form.
			C_1 = _mm_loadu_ps( finalC + offset1 );
			C_2 = _mm_loadu_ps( finalC + offset1+u );
			C_3 = _mm_loadu_ps( finalC + offset1+2*u );
			C_4 = _mm_loadu_ps( finalC + offset1+3*u );

			for ( k = 0 ; k < v; k+=blocksize )
			{

				offset2 = j+k*u;
				//load input matrix variables in vector form.
				A_1 = _mm_loadu_ps( finalA + i+k*u );
				A_2 = _mm_loadu_ps( finalA + i+(k+1)*u);
				A_3 = _mm_loadu_ps( finalA + i+(k+2)*u);
				A_4 = _mm_loadu_ps( finalA + i+(k+3)*u);

				//load the first row of the transposed input matrix variables in identical vector form.
				B_1 = _mm_load1_ps( finalA + offset2 );
				B_2 = _mm_load1_ps( finalA + offset2+u );
				B_3 = _mm_load1_ps( finalA + offset2+2*u );
				B_4 = _mm_load1_ps( finalA + offset2+3*u );

				C_1 = _mm_add_ps( C_1, _mm_mul_ps( A_1, B_1 ) );
				C_1 = _mm_add_ps( C_1, _mm_mul_ps( A_2, B_2 ) );
				C_1 = _mm_add_ps( C_1, _mm_mul_ps( A_3, B_3 ) );
				C_1 = _mm_add_ps( C_1, _mm_mul_ps( A_4, B_4 ) );

				//load the second row of the transposed input matrix variables in identical vector form.
				B_1 = _mm_load1_ps( finalA + offset2+1 );
				B_2 = _mm_load1_ps( finalA + offset2+u+1 );
				B_3 = _mm_load1_ps( finalA + offset2+2*u+1 );
				B_4 = _mm_load1_ps( finalA + offset2+3*u+1 );

				C_2 = _mm_add_ps( C_2, _mm_mul_ps( A_1, B_1 ) );
				C_2 = _mm_add_ps( C_2, _mm_mul_ps( A_2, B_2 ) );
				C_2 = _mm_add_ps( C_2, _mm_mul_ps( A_3, B_3 ) );
				C_2 = _mm_add_ps( C_2, _mm_mul_ps( A_4, B_4 ) );

				//load the third row of the transposed input matrix variables in identical vector form.
				B_1 = _mm_load1_ps( finalA + offset2+2 );
				B_2 = _mm_load1_ps( finalA + offset2+u+2 );
				B_3 = _mm_load1_ps( finalA + offset2+2*u+2 );
				B_4 = _mm_load1_ps( finalA + offset2+3*u+2 );

				C_3 = _mm_add_ps( C_3, _mm_mul_ps( A_1, B_1 ) );
				C_3 = _mm_add_ps( C_3, _mm_mul_ps( A_2, B_2 ) );
				C_3 = _mm_add_ps( C_3, _mm_mul_ps( A_3, B_3 ) );
				C_3 = _mm_add_ps( C_3, _mm_mul_ps( A_4, B_4 ) );

				//load the third row of the transposed input matrix variables in identical vector form.
				B_1 = _mm_load1_ps( finalA + offset2+3);
				B_2 = _mm_load1_ps( finalA + offset2+u+3 );
				B_3 = _mm_load1_ps( finalA + offset2+2*u+3 );
				B_4 = _mm_load1_ps( finalA + offset2+3*u+3 );

				C_4 = _mm_add_ps( C_4, _mm_mul_ps( A_1, B_1 ) );
				C_4 = _mm_add_ps( C_4, _mm_mul_ps( A_2, B_2 ) );
				C_4 = _mm_add_ps( C_4, _mm_mul_ps( A_3, B_3 ) );
				C_4 = _mm_add_ps( C_4, _mm_mul_ps( A_4, B_4 ) );
			}

			//store final matrix valus in vector form.
			_mm_storeu_ps( finalC + offset1, C_1 );
			_mm_storeu_ps( finalC + offset1+u, C_2 );
			_mm_storeu_ps( finalC + offset1+2*u, C_3 );
			_mm_storeu_ps( finalC + offset1+3*u, C_4 );

		}

	}


	//Truncating.
	if ( u != m )
	{
		for ( j = 0; j < m; j++ )
		{
			bound = m/10*10;
			offset1 = j*m;
			offset2 = j*u;
			for ( i = 0; i < bound; i+=10 )
			{
				C[offset1] = finalC[offset2];
				C[offset1+1] = finalC[offset2+1];
				C[offset1+2] = finalC[offset2+2];
				C[offset1+3] = finalC[offset2+3];
				C[offset1+4] = finalC[offset2+4];
				C[offset1+5] = finalC[offset2+5];
				C[offset1+6] = finalC[offset2+6];
				C[offset1+7] = finalC[offset2+7];
				C[offset1+8] = finalC[offset2+8];
				C[offset1+9] = finalC[offset2+9];
				offset1 += 10;
				offset2 += 10;
			}
			for ( i = bound; i < m; i++ )
			{
				C[i+j*m] = finalC[i+j*u];
			}
		}
		free(finalC);
		free(finalA);
	}
	else
	{
		if (v != n)
		{
			free(finalA);
		}
	}

}
