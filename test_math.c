#include <emmintrin.h>
#include <stdio.h>

int main(int argc, char * argv[])
{
    short A[8] = {2,4,6,8,10,8,6,4};
    short B[8] = {1,2,3,4,5,6,7,8};
    short R[8] = {0};
    
    __m128i xmmA = _mm_load_si128((__m128i*)A);
    __m128i xmmB = _mm_load_si128((__m128i*)B);

    __m128i xmmR = _mm_add_epi16(xmmA, xmmB);

    _mm_store_si128 ((__m128i*)R, xmmR);
    printf("Add : %d, %d, %d, %d, %d, %d, %d, %d\n",
            R[7], R[6], R[5], R[4], 
            R[3], R[2], R[1], R[0]);
    return 0;
}
