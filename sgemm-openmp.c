#include <stdlib.h>
#include <emmintrin.h>
#include <pmmintrin.h>

void sgemm( int m, int n, float *A, float *C )
{
	register __m128 B_1, B_2, B_3, B_4, B_5, B_6, B_7;
	register __m128 C_1, C_2, C_3, C_4;
	float *offset1, *offset2, *offset3;
	int i, j, k;
	int s = n%7;
	int m2 = m*2, m3 = m*3, m4 = m*4, m5 = m*5, m6 = m*6;


	# pragma omp parallel for schedule(guided, 1) private(i, j, k, offset1, offset2, offset3, B_1, B_2, B_3, B_4, B_5, B_6, B_7, C_1, C_2, C_3, C_4)
	for( j = 0; j < m; j++ ){
		for(k = 0; k < n/7*7; k+=7 ){
			offset1 = C + j * m;
			offset2 = A + k * m;
			offset3 = j + offset2;
			B_1 = _mm_load1_ps(offset3);
			B_2 = _mm_load1_ps(offset3+m);
			B_3 = _mm_load1_ps(offset3+m2);
			B_4 = _mm_load1_ps(offset3+m3);
			B_5 = _mm_load1_ps(offset3+m4);
			B_6 = _mm_load1_ps(offset3+m5);
			B_7 = _mm_load1_ps(offset3+m6);

			for(i=0; i < m/32*32; i+=32 ){
				C_1 = _mm_loadu_ps(offset1);
				C_2 = _mm_loadu_ps(offset1+4);
				C_3 = _mm_loadu_ps(offset1+8);
				C_4 = _mm_loadu_ps(offset1+12);

				C_1 = _mm_add_ps(C_1, _mm_mul_ps(_mm_loadu_ps(offset2), B_1));
				C_2 = _mm_add_ps(C_2, _mm_mul_ps(_mm_loadu_ps(offset2+4), B_1));
				C_3 = _mm_add_ps(C_3, _mm_mul_ps(_mm_loadu_ps(offset2+8), B_1));
				C_4 = _mm_add_ps(C_4, _mm_mul_ps(_mm_loadu_ps(offset2+12), B_1));

				offset2 += m;
				C_1 = _mm_add_ps(C_1, _mm_mul_ps(_mm_loadu_ps(offset2), B_2));
				C_2 = _mm_add_ps(C_2, _mm_mul_ps(_mm_loadu_ps(offset2+4), B_2));
				C_3 = _mm_add_ps(C_3, _mm_mul_ps(_mm_loadu_ps(offset2+8), B_2));
				C_4 = _mm_add_ps(C_4, _mm_mul_ps(_mm_loadu_ps(offset2+12), B_2));

				offset2 += m;
				C_1 = _mm_add_ps(C_1, _mm_mul_ps(_mm_loadu_ps(offset2), B_3));
				C_2 = _mm_add_ps(C_2, _mm_mul_ps(_mm_loadu_ps(offset2+4), B_3));
				C_3 = _mm_add_ps(C_3, _mm_mul_ps(_mm_loadu_ps(offset2+8), B_3));
				C_4 = _mm_add_ps(C_4, _mm_mul_ps(_mm_loadu_ps(offset2+12), B_3));

				offset2 += m;
				C_1 = _mm_add_ps(C_1, _mm_mul_ps(_mm_loadu_ps(offset2), B_4));
				C_2 = _mm_add_ps(C_2, _mm_mul_ps(_mm_loadu_ps(offset2+4), B_4));
				C_3 = _mm_add_ps(C_3, _mm_mul_ps(_mm_loadu_ps(offset2+8), B_4));
				C_4 = _mm_add_ps(C_4, _mm_mul_ps(_mm_loadu_ps(offset2+12), B_4));

				offset2 += m;
				C_1 = _mm_add_ps(C_1, _mm_mul_ps(_mm_loadu_ps(offset2), B_5));
				C_2 = _mm_add_ps(C_2, _mm_mul_ps(_mm_loadu_ps(offset2+4), B_5));
				C_3 = _mm_add_ps(C_3, _mm_mul_ps(_mm_loadu_ps(offset2+8), B_5));
				C_4 = _mm_add_ps(C_4, _mm_mul_ps(_mm_loadu_ps(offset2+12), B_5));

				offset2 += m;
				C_1 = _mm_add_ps(C_1, _mm_mul_ps(_mm_loadu_ps(offset2), B_6));
				C_2 = _mm_add_ps(C_2, _mm_mul_ps(_mm_loadu_ps(offset2+4), B_6));
				C_3 = _mm_add_ps(C_3, _mm_mul_ps(_mm_loadu_ps(offset2+8), B_6));
				C_4 = _mm_add_ps(C_4, _mm_mul_ps(_mm_loadu_ps(offset2+12), B_6));

				offset2 += m;
				C_1 = _mm_add_ps(C_1, _mm_mul_ps(_mm_loadu_ps(offset2), B_7));
				C_2 = _mm_add_ps(C_2, _mm_mul_ps(_mm_loadu_ps(offset2+4), B_7));
				C_3 = _mm_add_ps(C_3, _mm_mul_ps(_mm_loadu_ps(offset2+8), B_7));
				C_4 = _mm_add_ps(C_4, _mm_mul_ps(_mm_loadu_ps(offset2+12), B_7));

				_mm_storeu_ps(offset1, C_1);
				_mm_storeu_ps(offset1 + 4, C_2);
				_mm_storeu_ps(offset1 + 8, C_3);
				_mm_storeu_ps(offset1 + 12, C_4);

				offset1+=16;
				offset2+=16;

				C_1 = _mm_loadu_ps(offset1);
				C_2 = _mm_loadu_ps(offset1+4);
				C_3 = _mm_loadu_ps(offset1+8);
				C_4 = _mm_loadu_ps(offset1+12);

				C_1 = _mm_add_ps(C_1, _mm_mul_ps(_mm_loadu_ps(offset2), B_7));
				C_2 = _mm_add_ps(C_2, _mm_mul_ps(_mm_loadu_ps(offset2+4), B_7));
				C_3 = _mm_add_ps(C_3, _mm_mul_ps(_mm_loadu_ps(offset2+8), B_7));
				C_4 = _mm_add_ps(C_4, _mm_mul_ps(_mm_loadu_ps(offset2+12), B_7));

				offset2 -= m;
				C_1 = _mm_add_ps(C_1, _mm_mul_ps(_mm_loadu_ps(offset2), B_6));
				C_2 = _mm_add_ps(C_2, _mm_mul_ps(_mm_loadu_ps(offset2+4), B_6));
				C_3 = _mm_add_ps(C_3, _mm_mul_ps(_mm_loadu_ps(offset2+8), B_6));
				C_4 = _mm_add_ps(C_4, _mm_mul_ps(_mm_loadu_ps(offset2+12), B_6));

				offset2 -= m;
				C_1 = _mm_add_ps(C_1, _mm_mul_ps(_mm_loadu_ps(offset2), B_5));
				C_2 = _mm_add_ps(C_2, _mm_mul_ps(_mm_loadu_ps(offset2+4), B_5));
				C_3 = _mm_add_ps(C_3, _mm_mul_ps(_mm_loadu_ps(offset2+8), B_5));
				C_4 = _mm_add_ps(C_4, _mm_mul_ps(_mm_loadu_ps(offset2+12), B_5));

				offset2 -= m;
				C_1 = _mm_add_ps(C_1, _mm_mul_ps(_mm_loadu_ps(offset2), B_4));
				C_2 = _mm_add_ps(C_2, _mm_mul_ps(_mm_loadu_ps(offset2+4), B_4));
				C_3 = _mm_add_ps(C_3, _mm_mul_ps(_mm_loadu_ps(offset2+8), B_4));
				C_4 = _mm_add_ps(C_4, _mm_mul_ps(_mm_loadu_ps(offset2+12), B_4));

				offset2 -= m;
				C_1 = _mm_add_ps(C_1, _mm_mul_ps(_mm_loadu_ps(offset2), B_3));
				C_2 = _mm_add_ps(C_2, _mm_mul_ps(_mm_loadu_ps(offset2+4), B_3));
				C_3 = _mm_add_ps(C_3, _mm_mul_ps(_mm_loadu_ps(offset2+8), B_3));
				C_4 = _mm_add_ps(C_4, _mm_mul_ps(_mm_loadu_ps(offset2+12), B_3));

				offset2 -= m;
				C_1 = _mm_add_ps(C_1, _mm_mul_ps(_mm_loadu_ps(offset2), B_2));
				C_2 = _mm_add_ps(C_2, _mm_mul_ps(_mm_loadu_ps(offset2+4), B_2));
				C_3 = _mm_add_ps(C_3, _mm_mul_ps(_mm_loadu_ps(offset2+8), B_2));
				C_4 = _mm_add_ps(C_4, _mm_mul_ps(_mm_loadu_ps(offset2+12), B_2));

				offset2 -= m;
				C_1 = _mm_add_ps(C_1, _mm_mul_ps(_mm_loadu_ps(offset2), B_1));
				C_2 = _mm_add_ps(C_2, _mm_mul_ps(_mm_loadu_ps(offset2+4), B_1));
				C_3 = _mm_add_ps(C_3, _mm_mul_ps(_mm_loadu_ps(offset2+8), B_1));
				C_4 = _mm_add_ps(C_4, _mm_mul_ps(_mm_loadu_ps(offset2+12), B_1));

				_mm_storeu_ps(offset1, C_1);
				_mm_storeu_ps(offset1 + 4, C_2);
				_mm_storeu_ps(offset1 + 8, C_3);
				_mm_storeu_ps(offset1 + 12, C_4);

				offset1+=16;
				offset2+=16;

			}


			for(; i < m/16*16; i+=16 ){
				C_1 = _mm_loadu_ps(offset1);
				C_2 = _mm_loadu_ps(offset1+4);
				C_3 = _mm_loadu_ps(offset1+8);
				C_4 = _mm_loadu_ps(offset1+12);

				C_1 = _mm_add_ps(C_1, _mm_mul_ps(_mm_loadu_ps(offset2), B_1));
				C_2 = _mm_add_ps(C_2, _mm_mul_ps(_mm_loadu_ps(offset2+4), B_1));
				C_3 = _mm_add_ps(C_3, _mm_mul_ps(_mm_loadu_ps(offset2+8), B_1));
				C_4 = _mm_add_ps(C_4, _mm_mul_ps(_mm_loadu_ps(offset2+12), B_1));

				offset2 += m;
				C_1 = _mm_add_ps(C_1, _mm_mul_ps(_mm_loadu_ps(offset2), B_2));
				C_2 = _mm_add_ps(C_2, _mm_mul_ps(_mm_loadu_ps(offset2+4), B_2));
				C_3 = _mm_add_ps(C_3, _mm_mul_ps(_mm_loadu_ps(offset2+8), B_2));
				C_4 = _mm_add_ps(C_4, _mm_mul_ps(_mm_loadu_ps(offset2+12), B_2));

				offset2 += m;
				C_1 = _mm_add_ps(C_1, _mm_mul_ps(_mm_loadu_ps(offset2), B_3));
				C_2 = _mm_add_ps(C_2, _mm_mul_ps(_mm_loadu_ps(offset2+4), B_3));
				C_3 = _mm_add_ps(C_3, _mm_mul_ps(_mm_loadu_ps(offset2+8), B_3));
				C_4 = _mm_add_ps(C_4, _mm_mul_ps(_mm_loadu_ps(offset2+12), B_3));

				offset2 += m;
				C_1 = _mm_add_ps(C_1, _mm_mul_ps(_mm_loadu_ps(offset2), B_4));
				C_2 = _mm_add_ps(C_2, _mm_mul_ps(_mm_loadu_ps(offset2+4), B_4));
				C_3 = _mm_add_ps(C_3, _mm_mul_ps(_mm_loadu_ps(offset2+8), B_4));
				C_4 = _mm_add_ps(C_4, _mm_mul_ps(_mm_loadu_ps(offset2+12), B_4));

				offset2 += m;
				C_1 = _mm_add_ps(C_1, _mm_mul_ps(_mm_loadu_ps(offset2), B_5));
				C_2 = _mm_add_ps(C_2, _mm_mul_ps(_mm_loadu_ps(offset2+4), B_5));
				C_3 = _mm_add_ps(C_3, _mm_mul_ps(_mm_loadu_ps(offset2+8), B_5));
				C_4 = _mm_add_ps(C_4, _mm_mul_ps(_mm_loadu_ps(offset2+12), B_5));

				offset2 += m;
				C_1 = _mm_add_ps(C_1, _mm_mul_ps(_mm_loadu_ps(offset2), B_6));
				C_2 = _mm_add_ps(C_2, _mm_mul_ps(_mm_loadu_ps(offset2+4), B_6));
				C_3 = _mm_add_ps(C_3, _mm_mul_ps(_mm_loadu_ps(offset2+8), B_6));
				C_4 = _mm_add_ps(C_4, _mm_mul_ps(_mm_loadu_ps(offset2+12), B_6));

				offset2 += m;
				C_1 = _mm_add_ps(C_1, _mm_mul_ps(_mm_loadu_ps(offset2), B_7));
				C_2 = _mm_add_ps(C_2, _mm_mul_ps(_mm_loadu_ps(offset2+4), B_7));
				C_3 = _mm_add_ps(C_3, _mm_mul_ps(_mm_loadu_ps(offset2+8), B_7));
				C_4 = _mm_add_ps(C_4, _mm_mul_ps(_mm_loadu_ps(offset2+12), B_7));

				_mm_storeu_ps(offset1, C_1);
				_mm_storeu_ps(offset1 + 4, C_2);
				_mm_storeu_ps(offset1 + 8, C_3);
				_mm_storeu_ps(offset1 + 12, C_4);

				offset1+=16;
				offset2=offset2-m6+16;
			}
			for(; i < m/4*4; i+=4 ){
				C_1 = _mm_loadu_ps(offset1);
				C_1 = _mm_add_ps(C_1, _mm_mul_ps(_mm_loadu_ps(offset2), B_1));
				C_1 = _mm_add_ps(C_1, _mm_mul_ps(_mm_loadu_ps(offset2 + m), B_2));
				C_1 = _mm_add_ps(C_1, _mm_mul_ps(_mm_loadu_ps(offset2 + m2), B_3));
				C_1 = _mm_add_ps(C_1, _mm_mul_ps(_mm_loadu_ps(offset2 + m3), B_4));
				C_1 = _mm_add_ps(C_1, _mm_mul_ps(_mm_loadu_ps(offset2 + m4), B_5));
				C_1 = _mm_add_ps(C_1, _mm_mul_ps(_mm_loadu_ps(offset2 + m5), B_6));
				C_1 = _mm_add_ps(C_1, _mm_mul_ps(_mm_loadu_ps(offset2 + m6), B_7));
				_mm_storeu_ps(offset1, C_1);
				offset2+=4;
				offset1+=4;
			}
			for(; i < m; i++ ){
				*(offset1) += (*(offset2)) * (*offset3)
					+ (*(offset2+m)) * (*(offset3+m))
					+ (*(offset2+m2)) * (*(offset3+m2))
					+ (*(offset2+m3)) * (*(offset3+m3))
					+ (*(offset2+m4)) * (*(offset3+m4))
					+ (*(offset2+m5)) * (*(offset3+m5))
					+ (*(offset2+m6)) * (*(offset3+m6));
				offset1++;
				offset2++;
			}
		}

		offset1 = C + j * m;
		offset2 = A + k * m;
		offset3 = j + offset2;
		switch(s){
			case 0:
			break;

			case 1:
			B_1 = _mm_load1_ps(offset3);
			for( i = 0; i < m/16*16; i+=16 ){
				C_1 = _mm_loadu_ps(offset1);
				C_2 = _mm_loadu_ps(offset1+4);
				C_3 = _mm_loadu_ps(offset1+8);
				C_4 = _mm_loadu_ps(offset1+12);

				C_1 = _mm_add_ps(C_1, _mm_mul_ps(_mm_loadu_ps(offset2), B_1));
				C_2 = _mm_add_ps(C_2, _mm_mul_ps(_mm_loadu_ps(offset2+4), B_1));
				C_3 = _mm_add_ps(C_3, _mm_mul_ps(_mm_loadu_ps(offset2+8), B_1));
				C_4 = _mm_add_ps(C_4, _mm_mul_ps(_mm_loadu_ps(offset2+12), B_1));
				_mm_storeu_ps(offset1, C_1);
				_mm_storeu_ps(offset1 + 4, C_2);
				_mm_storeu_ps(offset1 + 8, C_3);
				_mm_storeu_ps(offset1 + 12, C_4);
				offset1 += 16;
				offset2 += 16;
			}
			for(; i < m/4*4; i+=4 ){
				C_1 = _mm_loadu_ps(offset1);
				C_1 = _mm_add_ps(C_1, _mm_mul_ps(_mm_loadu_ps(offset2), B_1));
				_mm_storeu_ps(offset1, C_1);
				offset1+=4;
				offset2+=4;
			}
			for(; i < m; i++ ){
				*(offset1) += (*(offset2)) * (*offset3);
				offset1++;
				offset2++;
			}
			break;
			case 2:
			B_1 = _mm_load1_ps(offset3);
			B_2 = _mm_load1_ps(offset3+m);
			for( i = 0; i < m/16*16; i+=16 ){
				C_1 = _mm_loadu_ps(offset1);
				C_2 = _mm_loadu_ps(offset1+4);
				C_3 = _mm_loadu_ps(offset1+8);
				C_4 = _mm_loadu_ps(offset1+12);

				C_1 = _mm_add_ps(C_1, _mm_mul_ps(_mm_loadu_ps(offset2), B_1));
				C_2 = _mm_add_ps(C_2, _mm_mul_ps(_mm_loadu_ps(offset2+4), B_1));
				C_3 = _mm_add_ps(C_3, _mm_mul_ps(_mm_loadu_ps(offset2+8), B_1));
				C_4 = _mm_add_ps(C_4, _mm_mul_ps(_mm_loadu_ps(offset2+12), B_1));

				offset2 += m;
				C_1 = _mm_add_ps(C_1, _mm_mul_ps(_mm_loadu_ps(offset2), B_2));
				C_2 = _mm_add_ps(C_2, _mm_mul_ps(_mm_loadu_ps(offset2+4), B_2));
				C_3 = _mm_add_ps(C_3, _mm_mul_ps(_mm_loadu_ps(offset2+8), B_2));
				C_4 = _mm_add_ps(C_4, _mm_mul_ps(_mm_loadu_ps(offset2+12), B_2));

				_mm_storeu_ps(offset1, C_1);
				_mm_storeu_ps(offset1 + 4, C_2);
				_mm_storeu_ps(offset1 + 8, C_3);
				_mm_storeu_ps(offset1 + 12, C_4);
				offset1 += 16;
				offset2 = offset2 + 16 - m;
			}
			for(; i < m/4*4; i+=4 ){
				C_1 = _mm_loadu_ps(offset1);
				C_1 = _mm_add_ps(C_1, _mm_mul_ps(_mm_loadu_ps(offset2), B_1));
				C_1 = _mm_add_ps(C_1, _mm_mul_ps(_mm_loadu_ps(offset2 + m), B_2));
				_mm_storeu_ps(offset1, C_1);
				offset1+=4;
				offset2+=4;
			}
			for(; i < m; i++ ){
				*(offset1) += (*(offset2)) * (*offset3) +
						+ (*(offset2+m)) * (*(offset3+m));
				offset1++;
				offset2++;
			}
			break;

			case 3:
			B_1 = _mm_load1_ps(offset3);
			B_2 = _mm_load1_ps(offset3+m);
			B_3 = _mm_load1_ps(offset3+m2);
			for( i = 0; i < m/16*16; i+=16 ){
				C_1 = _mm_loadu_ps(offset1);
				C_2 = _mm_loadu_ps(offset1+4);
				C_3 = _mm_loadu_ps(offset1+8);
				C_4 = _mm_loadu_ps(offset1+12);

				C_1 = _mm_add_ps(C_1, _mm_mul_ps(_mm_loadu_ps(offset2), B_1));
				C_2 = _mm_add_ps(C_2, _mm_mul_ps(_mm_loadu_ps(offset2+4), B_1));
				C_3 = _mm_add_ps(C_3, _mm_mul_ps(_mm_loadu_ps(offset2+8), B_1));
				C_4 = _mm_add_ps(C_4, _mm_mul_ps(_mm_loadu_ps(offset2+12), B_1));

				offset2 += m;
				C_1 = _mm_add_ps(C_1, _mm_mul_ps(_mm_loadu_ps(offset2), B_2));
				C_2 = _mm_add_ps(C_2, _mm_mul_ps(_mm_loadu_ps(offset2+4), B_2));
				C_3 = _mm_add_ps(C_3, _mm_mul_ps(_mm_loadu_ps(offset2+8), B_2));
				C_4 = _mm_add_ps(C_4, _mm_mul_ps(_mm_loadu_ps(offset2+12), B_2));

				offset2 += m;
				C_1 = _mm_add_ps(C_1, _mm_mul_ps(_mm_loadu_ps(offset2), B_3));
				C_2 = _mm_add_ps(C_2, _mm_mul_ps(_mm_loadu_ps(offset2+4), B_3));
				C_3 = _mm_add_ps(C_3, _mm_mul_ps(_mm_loadu_ps(offset2+8), B_3));
				C_4 = _mm_add_ps(C_4, _mm_mul_ps(_mm_loadu_ps(offset2+12), B_3));

				_mm_storeu_ps(offset1, C_1);
				_mm_storeu_ps(offset1 + 4, C_2);
				_mm_storeu_ps(offset1 + 8, C_3);
				_mm_storeu_ps(offset1 + 12, C_4);
				offset1 += 16;
				offset2 = offset2 + 16 - m2;
			}
			for(; i < m/4*4; i+=4 ){
				C_1 = _mm_loadu_ps(offset1);
				C_1 = _mm_add_ps(C_1, _mm_mul_ps(_mm_loadu_ps(offset2), B_1));
				C_1 = _mm_add_ps(C_1, _mm_mul_ps(_mm_loadu_ps(offset2 + m), B_2));
				C_1 = _mm_add_ps(C_1, _mm_mul_ps(_mm_loadu_ps(offset2 + m2), B_3));
				_mm_storeu_ps(offset1, C_1);
				offset1+=4;
				offset2+=4;
			}
			for(; i < m; i++ ){
				*(offset1) += (*(offset2)) * (*offset3) +
						+ (*(offset2+m)) * (*(offset3+m))
						+ (*(offset2+m2)) * (*(offset3+m2));
				offset1++;
				offset2++;
			}
			break;


			case 4:
			B_1 = _mm_load1_ps(offset3);
			B_2 = _mm_load1_ps(offset3+m);
			B_3 = _mm_load1_ps(offset3+m2);
			B_4 = _mm_load1_ps(offset3+m3);
			for( i = 0; i < m/16*16; i+=16 ){
				C_1 = _mm_loadu_ps(offset1);
				C_2 = _mm_loadu_ps(offset1+4);
				C_3 = _mm_loadu_ps(offset1+8);
				C_4 = _mm_loadu_ps(offset1+12);

				C_1 = _mm_add_ps(C_1, _mm_mul_ps(_mm_loadu_ps(offset2), B_1));
				C_2 = _mm_add_ps(C_2, _mm_mul_ps(_mm_loadu_ps(offset2+4), B_1));
				C_3 = _mm_add_ps(C_3, _mm_mul_ps(_mm_loadu_ps(offset2+8), B_1));
				C_4 = _mm_add_ps(C_4, _mm_mul_ps(_mm_loadu_ps(offset2+12), B_1));

				offset2 += m;
				C_1 = _mm_add_ps(C_1, _mm_mul_ps(_mm_loadu_ps(offset2), B_2));
				C_2 = _mm_add_ps(C_2, _mm_mul_ps(_mm_loadu_ps(offset2+4), B_2));
				C_3 = _mm_add_ps(C_3, _mm_mul_ps(_mm_loadu_ps(offset2+8), B_2));
				C_4 = _mm_add_ps(C_4, _mm_mul_ps(_mm_loadu_ps(offset2+12), B_2));

				offset2 += m;
				C_1 = _mm_add_ps(C_1, _mm_mul_ps(_mm_loadu_ps(offset2), B_3));
				C_2 = _mm_add_ps(C_2, _mm_mul_ps(_mm_loadu_ps(offset2+4), B_3));
				C_3 = _mm_add_ps(C_3, _mm_mul_ps(_mm_loadu_ps(offset2+8), B_3));
				C_4 = _mm_add_ps(C_4, _mm_mul_ps(_mm_loadu_ps(offset2+12), B_3));

				offset2 += m;
				C_1 = _mm_add_ps(C_1, _mm_mul_ps(_mm_loadu_ps(offset2), B_4));
				C_2 = _mm_add_ps(C_2, _mm_mul_ps(_mm_loadu_ps(offset2+4), B_4));
				C_3 = _mm_add_ps(C_3, _mm_mul_ps(_mm_loadu_ps(offset2+8), B_4));
				C_4 = _mm_add_ps(C_4, _mm_mul_ps(_mm_loadu_ps(offset2+12), B_4));


				_mm_storeu_ps(offset1, C_1);
				_mm_storeu_ps(offset1 + 4, C_2);
				_mm_storeu_ps(offset1 + 8, C_3);
				_mm_storeu_ps(offset1 + 12, C_4);
				offset1 += 16;
				offset2 = offset2 + 16 - m3;
			}
			for(; i < m/4*4; i+=4 ){
				C_1 = _mm_loadu_ps(offset1);
				C_1 = _mm_add_ps(C_1, _mm_mul_ps(_mm_loadu_ps(offset2), B_1));
				C_1 = _mm_add_ps(C_1, _mm_mul_ps(_mm_loadu_ps(offset2 + m), B_2));
				C_1 = _mm_add_ps(C_1, _mm_mul_ps(_mm_loadu_ps(offset2 + m2), B_3));
				C_1 = _mm_add_ps(C_1, _mm_mul_ps(_mm_loadu_ps(offset2 + m3), B_4));
				_mm_storeu_ps(offset1, C_1);
				offset1+=4;
				offset2+=4;
			}
			for(; i < m; i++ ){
				*(offset1) += (*(offset2)) * (*offset3) +
						+ (*(offset2+m)) * (*(offset3+m))
						+ (*(offset2+m2)) * (*(offset3+m2))
						+ (*(offset2+m3)) * (*(offset3+m3));
				offset1++;
				offset2++;
			}
			break;

			case 5:
			B_1 = _mm_load1_ps(offset3);
			B_2 = _mm_load1_ps(offset3+m);
			B_3 = _mm_load1_ps(offset3+m2);
			B_4 = _mm_load1_ps(offset3+m3);
			B_5 = _mm_load1_ps(offset3+m4);
			for( i = 0; i < m/16*16; i+=16 ){
				C_1 = _mm_loadu_ps(offset1);
				C_2 = _mm_loadu_ps(offset1+4);
				C_3 = _mm_loadu_ps(offset1+8);
				C_4 = _mm_loadu_ps(offset1+12);

				C_1 = _mm_add_ps(C_1, _mm_mul_ps(_mm_loadu_ps(offset2), B_1));
				C_2 = _mm_add_ps(C_2, _mm_mul_ps(_mm_loadu_ps(offset2+4), B_1));
				C_3 = _mm_add_ps(C_3, _mm_mul_ps(_mm_loadu_ps(offset2+8), B_1));
				C_4 = _mm_add_ps(C_4, _mm_mul_ps(_mm_loadu_ps(offset2+12), B_1));

				offset2 += m;
				C_1 = _mm_add_ps(C_1, _mm_mul_ps(_mm_loadu_ps(offset2), B_2));
				C_2 = _mm_add_ps(C_2, _mm_mul_ps(_mm_loadu_ps(offset2+4), B_2));
				C_3 = _mm_add_ps(C_3, _mm_mul_ps(_mm_loadu_ps(offset2+8), B_2));
				C_4 = _mm_add_ps(C_4, _mm_mul_ps(_mm_loadu_ps(offset2+12), B_2));

				offset2 += m;
				C_1 = _mm_add_ps(C_1, _mm_mul_ps(_mm_loadu_ps(offset2), B_3));
				C_2 = _mm_add_ps(C_2, _mm_mul_ps(_mm_loadu_ps(offset2+4), B_3));
				C_3 = _mm_add_ps(C_3, _mm_mul_ps(_mm_loadu_ps(offset2+8), B_3));
				C_4 = _mm_add_ps(C_4, _mm_mul_ps(_mm_loadu_ps(offset2+12), B_3));

				offset2 += m;
				C_1 = _mm_add_ps(C_1, _mm_mul_ps(_mm_loadu_ps(offset2), B_4));
				C_2 = _mm_add_ps(C_2, _mm_mul_ps(_mm_loadu_ps(offset2+4), B_4));
				C_3 = _mm_add_ps(C_3, _mm_mul_ps(_mm_loadu_ps(offset2+8), B_4));
				C_4 = _mm_add_ps(C_4, _mm_mul_ps(_mm_loadu_ps(offset2+12), B_4));

				offset2 += m;
				C_1 = _mm_add_ps(C_1, _mm_mul_ps(_mm_loadu_ps(offset2), B_5));
				C_2 = _mm_add_ps(C_2, _mm_mul_ps(_mm_loadu_ps(offset2+4), B_5));
				C_3 = _mm_add_ps(C_3, _mm_mul_ps(_mm_loadu_ps(offset2+8), B_5));
				C_4 = _mm_add_ps(C_4, _mm_mul_ps(_mm_loadu_ps(offset2+12), B_5));

				_mm_storeu_ps(offset1, C_1);
				_mm_storeu_ps(offset1 + 4, C_2);
				_mm_storeu_ps(offset1 + 8, C_3);
				_mm_storeu_ps(offset1 + 12, C_4);
				offset1 += 16;
				offset2 = offset2 + 16 - m4;
			}
			for(; i < m/4*4; i+=4 ){
				C_1 = _mm_loadu_ps(offset1);
				C_1 = _mm_add_ps(C_1, _mm_mul_ps(_mm_loadu_ps(offset2), B_1));
				C_1 = _mm_add_ps(C_1, _mm_mul_ps(_mm_loadu_ps(offset2 + m), B_2));
				C_1 = _mm_add_ps(C_1, _mm_mul_ps(_mm_loadu_ps(offset2 + m2), B_3));
				C_1 = _mm_add_ps(C_1, _mm_mul_ps(_mm_loadu_ps(offset2 + m3), B_4));
				C_1 = _mm_add_ps(C_1, _mm_mul_ps(_mm_loadu_ps(offset2 + m4), B_5));
				_mm_storeu_ps(offset1, C_1);
				offset1+=4;
				offset2+=4;
			}
			for(; i < m; i++ ){
				*(offset1) += (*(offset2)) * (*offset3) +
						+ (*(offset2+m)) * (*(offset3+m))
						+ (*(offset2+m2)) * (*(offset3+m2))
						+ (*(offset2+m3)) * (*(offset3+m3))
						+ (*(offset2+m4)) * (*(offset3+m4));
				offset1++;
				offset2++;
			}
			break;



			case 6:
			B_1 = _mm_load1_ps(offset3);
			B_2 = _mm_load1_ps(offset3+m);
			B_3 = _mm_load1_ps(offset3+m2);
			B_4 = _mm_load1_ps(offset3+m3);
			B_5 = _mm_load1_ps(offset3+m4);
			B_6 = _mm_load1_ps(offset3+m5);
			for( i = 0; i < m/16*16; i+=16 ){
				C_1 = _mm_loadu_ps(offset1);
				C_2 = _mm_loadu_ps(offset1+4);
				C_3 = _mm_loadu_ps(offset1+8);
				C_4 = _mm_loadu_ps(offset1+12);

				C_1 = _mm_add_ps(C_1, _mm_mul_ps(_mm_loadu_ps(offset2), B_1));
				C_2 = _mm_add_ps(C_2, _mm_mul_ps(_mm_loadu_ps(offset2+4), B_1));
				C_3 = _mm_add_ps(C_3, _mm_mul_ps(_mm_loadu_ps(offset2+8), B_1));
				C_4 = _mm_add_ps(C_4, _mm_mul_ps(_mm_loadu_ps(offset2+12), B_1));

				offset2 += m;
				C_1 = _mm_add_ps(C_1, _mm_mul_ps(_mm_loadu_ps(offset2), B_2));
				C_2 = _mm_add_ps(C_2, _mm_mul_ps(_mm_loadu_ps(offset2+4), B_2));
				C_3 = _mm_add_ps(C_3, _mm_mul_ps(_mm_loadu_ps(offset2+8), B_2));
				C_4 = _mm_add_ps(C_4, _mm_mul_ps(_mm_loadu_ps(offset2+12), B_2));

				offset2 += m;
				C_1 = _mm_add_ps(C_1, _mm_mul_ps(_mm_loadu_ps(offset2), B_3));
				C_2 = _mm_add_ps(C_2, _mm_mul_ps(_mm_loadu_ps(offset2+4), B_3));
				C_3 = _mm_add_ps(C_3, _mm_mul_ps(_mm_loadu_ps(offset2+8), B_3));
				C_4 = _mm_add_ps(C_4, _mm_mul_ps(_mm_loadu_ps(offset2+12), B_3));

				offset2 += m;
				C_1 = _mm_add_ps(C_1, _mm_mul_ps(_mm_loadu_ps(offset2), B_4));
				C_2 = _mm_add_ps(C_2, _mm_mul_ps(_mm_loadu_ps(offset2+4), B_4));
				C_3 = _mm_add_ps(C_3, _mm_mul_ps(_mm_loadu_ps(offset2+8), B_4));
				C_4 = _mm_add_ps(C_4, _mm_mul_ps(_mm_loadu_ps(offset2+12), B_4));

				offset2 += m;
				C_1 = _mm_add_ps(C_1, _mm_mul_ps(_mm_loadu_ps(offset2), B_5));
				C_2 = _mm_add_ps(C_2, _mm_mul_ps(_mm_loadu_ps(offset2+4), B_5));
				C_3 = _mm_add_ps(C_3, _mm_mul_ps(_mm_loadu_ps(offset2+8), B_5));
				C_4 = _mm_add_ps(C_4, _mm_mul_ps(_mm_loadu_ps(offset2+12), B_5));

				offset2 += m;
				C_1 = _mm_add_ps(C_1, _mm_mul_ps(_mm_loadu_ps(offset2), B_6));
				C_2 = _mm_add_ps(C_2, _mm_mul_ps(_mm_loadu_ps(offset2+4), B_6));
				C_3 = _mm_add_ps(C_3, _mm_mul_ps(_mm_loadu_ps(offset2+8), B_6));
				C_4 = _mm_add_ps(C_4, _mm_mul_ps(_mm_loadu_ps(offset2+12), B_6));

				_mm_storeu_ps(offset1, C_1);
				_mm_storeu_ps(offset1 + 4, C_2);
				_mm_storeu_ps(offset1 + 8, C_3);
				_mm_storeu_ps(offset1 + 12, C_4);
				offset1 += 16;
				offset2 = offset2 + 16 - m5;
			}
			for(; i < m/4*4; i+=4 ){
				C_1 = _mm_loadu_ps(offset1);
				C_1 = _mm_add_ps(C_1, _mm_mul_ps(_mm_loadu_ps(offset2), B_1));
				C_1 = _mm_add_ps(C_1, _mm_mul_ps(_mm_loadu_ps(offset2 + m), B_2));
				C_1 = _mm_add_ps(C_1, _mm_mul_ps(_mm_loadu_ps(offset2 + m2), B_3));
				C_1 = _mm_add_ps(C_1, _mm_mul_ps(_mm_loadu_ps(offset2 + m3), B_4));
				C_1 = _mm_add_ps(C_1, _mm_mul_ps(_mm_loadu_ps(offset2 + m4), B_5));
				C_1 = _mm_add_ps(C_1, _mm_mul_ps(_mm_loadu_ps(offset2 + m5), B_6));
				_mm_storeu_ps(offset1, C_1);
				offset1+=4;
				offset2+=4;
			}
			for(; i < m; i++ ){
				*(offset1) += (*(offset2)) * (*offset3) +
						+ (*(offset2+m)) * (*(offset3+m))
						+ (*(offset2+m2)) * (*(offset3+m2))
						+ (*(offset2+m3)) * (*(offset3+m3))
						+ (*(offset2+m4)) * (*(offset3+m4))
						+ (*(offset2+m5)) * (*(offset3+m5));
				offset1++;
				offset2++;
			}
			break;

		}
	}
}