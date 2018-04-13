#include <emmintrin.h>
#include <stdio.h>

int main (int argc, char * argv[])
{
    short Source[8] = {1,2,3,4,5,6,7,8};
    short Dest [8] = {0};

    __m128i xmmA = _mm_loadu_si128((__m128i*)Source);
    _mm_storeu_si128((__m128i*) Dest,xmmA);

    printf("Dest : %d, %d, %d, %d, %d, %d, %d, %d\n",
            Dest[7], Dest[6], Dest[5], Dest[4],
            Dest[3], Dest[2], Dest[1], Dest[0]);
    return 0;
}


