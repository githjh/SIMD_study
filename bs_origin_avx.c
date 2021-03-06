
// Copyright (c) 2007 Intel Corp.

// Black-Scholes
// Analytical method for calculating European Options
//
// 
// Reference Source: Options, Futures, and Other Derivatives, 3rd Edition, Prentice 
// Hall, John C. Hull,

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <sys/time.h>
#include <xmmintrin.h>
#include <immintrin.h>

#define TIME_DIFF(B,E) ((E.tv_sec - B.tv_sec) + (E.tv_usec - B.tv_usec)*0.000001)


//Precision to use for calculations
#define fptype float

#define NUM_RUNS 100 /* DO NOT change this value */

typedef struct OptionData_ {
        fptype sptprice;          // spot price
        fptype strike;     // strike price
        fptype rate;          // risk-free interest rate
        fptype divq;       // dividend rate
        fptype volatility;          // volatility
        fptype otime;          // time to maturity or option expiration in years 
                           //     (1yr = 1.0, 6mos = 0.5, 3mos = 0.25, ..., etc)  
        char OptionType;   // Option type.  "P"=PUT, "C"=CALL
        fptype divs;       // dividend vals (not used in this test)
        fptype DGrefval;   // DerivaGem Reference Value (not used in this test)
} OptionData;


typedef struct database_{
        fptype *sptprice;          // spot price
        fptype *strike;     // strike price
        fptype *rate;          // risk-free interest rate
        fptype *divq;       // dividend rate
        fptype *volatility;          // volatility
        fptype *otime;          // time to maturity or option expiration in years 
                           //     (1yr = 1.0, 6mos = 0.5, 3mos = 0.25, ..., etc)  
        char *OptionType;   // Option type.  "P"=PUT, "C"=CALL
        fptype *divs;       // dividend vals (not used in this test)
        fptype *DGrefval;   // DerivaGem Reference Value (not used in this test)
} database;


database data;
fptype *prices;
int numOptions;

int nThreads;

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
// Cumulative Normal Distribution Function
// See Hull, Section 11.8, P.243-244
#define inv_sqrt_2xPI 0.39894228040143270286

__m256 CNDF ( __m256 InputX ) 
{
    __m256i sign;

    __m256 OutputX;
    __m256 xInput;
    __m256 xNPrimeofX;
    __m256 expValues;
    __m256 xK2;
    __m256 xK2_2, xK2_3;
    __m256 xK2_4, xK2_5;
    __m256 xLocal, xLocal_1;
    __m256 xLocal_2, xLocal_3;

    int i;
    // Check for negative value of InputX
    /*    for ( i = 0; i < 8; i++){ 
      if (((float*)&InputX)[i] < 0.0) {
	((float*)&InputX)[i] = -((float*)&InputX)[i];
	((int*)&sign)[i] = 1;
      } 
      else 
	((int*)&sign)[i] = 0;
	}*/

    __m256 ZeroValue = _mm256_setzero_ps();
    ZeroValue = _mm256_sub_ps(ZeroValue, InputX);
    InputX = _mm256_max_ps(ZeroValue, InputX);
    
    //sign = _mm256_cvtps_epi32(_mm256_cmp_ps(ZeroValue, _mm256_setzero_ps(), _CMP_GT_OQ));
    sign = _mm256_cvtps_epi32(_mm256_and_ps(_mm256_cmp_ps(ZeroValue, _mm256_setzero_ps(), _CMP_GT_OQ), _mm256_set1_ps(1)));
    

    xInput = InputX;

    /*
    int l;
    for ( l = 0; l < 4; l++){
      printf("inputx %dth: %f\n",l, ((float*)&InputX)[l]);
    }*/

      
    // Compute NPrimeX term common to both four & six decimal accuracy calcs
    //expValues = exp(-0.5f * InputX * InputX);
    __m256 MulTerm;
    //MulHalf = _mm_set_ps1(-0.5);
    MulTerm = _mm256_mul_ps(InputX, InputX);
    MulTerm = _mm256_mul_ps(_mm256_set1_ps(-0.5), MulTerm);
    expValues = _mm256_set_ps (exp(((float*)&MulTerm)[7]),
			       exp(((float*)&MulTerm)[6]),
			       exp(((float*)&MulTerm)[5]),
			       exp(((float*)&MulTerm)[4]), 
			       exp(((float*)&MulTerm)[3]),
			       exp(((float*)&MulTerm)[2]),
			       exp(((float*)&MulTerm)[1]),
			       exp(((float*)&MulTerm)[0])
			       );
    

    xNPrimeofX = expValues;
    //xNPrimeofX = xNPrimeofX * inv_sqrt_2xPI;
    //    __m128 inv_sqrt_Term;
    //inv_sqrt_Term = _mm_set_ps1(inv_sqrt_2xPI);
    xNPrimeofX = _mm256_mul_ps(xNPrimeofX, _mm256_set1_ps(inv_sqrt_2xPI));


    //xK2 = 0.2316419 * xInput;
    //    __m128 Mul023;
    //Mul023 =  _mm_set_ps1(0.2316419);
    xK2 = _mm256_mul_ps( _mm256_set1_ps(0.2316419), xInput);

    //xK2 = 1.0 + xK2;
    //__m128 Add1;
    //Add1 = _mm_set_ps1(1.0);
    xK2 = _mm256_add_ps(_mm256_set1_ps(1.0), xK2);

    //xK2 = 1.0 / xK2;
    xK2 = _mm256_div_ps(_mm256_set1_ps(1.0), xK2);

    //xK2_2 = xK2 * xK2;
    xK2_2 = _mm256_mul_ps(xK2, xK2);

    //xK2_3 = xK2_2 * xK2;
    xK2_3 = _mm256_mul_ps(xK2_2, xK2);

    //xK2_4 = xK2_3 * xK2;
    xK2_4 = _mm256_mul_ps(xK2_3, xK2);

    //xK2_5 = xK2_4 * xK2;
    xK2_5 = _mm256_mul_ps(xK2_4, xK2);
    
    //xLocal_1 = xK2 * 0.319381530;
    //    __m128 Mul031;
    //Mul031 = _mm_set_ps1(0.319381530);
    xLocal_1 = _mm256_mul_ps(xK2, _mm256_set1_ps(0.319381530));

    //xLocal_2 = xK2_2 * (-0.356563782);
    //    __m128 Mul035;
    //Mul035 = _mm_set_ps1(-0.356563782);
    xLocal_2 = _mm256_mul_ps(xK2_2, _mm256_set1_ps(-0.356563782));

    //xLocal_3 = xK2_3 * 1.781477937;
    //    __m128 Mul178;
    //Mul178 = _mm_set_ps1(1.781477937);
    xLocal_3 = _mm256_mul_ps(xK2_3, _mm256_set1_ps(1.781477937));

    //xLocal_2 = xLocal_2 + xLocal_3;
    xLocal_2 = _mm256_add_ps(xLocal_2, xLocal_3);

    //xLocal_3 = xK2_4 * (-1.821255978);
    //    __m128 Mul182;
    //Mul182 = _mm_set_ps1(-1.821255978);
    xLocal_3 = _mm256_mul_ps(xK2_4, _mm256_set1_ps(-1.821255978));
    

    //xLocal_2 = xLocal_2 + xLocal_3;
    xLocal_2 = _mm256_add_ps(xLocal_2, xLocal_3);

    //xLocal_3 = xK2_5 * 1.330274429;
    //    __m128 Mul133;
    //Mul133 = _mm_set_ps1(1.330274429);
    xLocal_3 = _mm256_mul_ps(xK2_5, _mm256_set1_ps(1.330274429));
    
    //xLocal_2 = xLocal_2 + xLocal_3;
    xLocal_2 = _mm256_add_ps(xLocal_2, xLocal_3);

    //xLocal_1 = xLocal_2 + xLocal_1;
    xLocal_1 = _mm256_add_ps(xLocal_2, xLocal_1);

    //xLocal   = xLocal_1 * xNPrimeofX;
    xLocal = _mm256_mul_ps(xLocal_1, xNPrimeofX);
    
    //xLocal   = 1.0 - xLocal;
    xLocal = _mm256_sub_ps(_mm256_set1_ps(1.0), xLocal); //Add1 declared up there

    OutputX  = xLocal;
    
    i = 0;
    for ( i = 0; i < 8; i++){
      if (((int*)&sign)[i]) {
        ((float*)&OutputX)[i] = 1.0 - ((float*)&OutputX)[i];
      }
      //printf("CNDF %dth: %f\n", i, ((float*)&OutputX)[i]);
    }
    /* OutputX = _mm_sub_ps(_mm_cvtepi32_ps(sign), OutputX);
    ZeroValue = _mm_setzero_ps();
    ZeroValue = _mm_sub_ps(ZeroValue, OutputX);
    OutputX = _mm_max_ps(ZeroValue, OutputX);
    */
    return OutputX;
} 

//////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////
__m256 BlkSchlsEqEuroNoDiv( __m256 sptprice,
                            __m256 strike, __m256 rate, __m256 volatility,
                            __m256 time, __m256i otype )
{
    __m256 OptionPrice;

    // local private working variables for the calculation
    __m256 xStockPrice;
    __m256 xStrikePrice;
    __m256 xRiskFreeRate;
    __m256 xVolatility;
    __m256 xTime;
    __m256 xSqrtTime;

    __m256 logValues;
    __m256 xLogTerm;
    __m256 xD1; 
    __m256 xD2;
    __m256 xPowerTerm;
    __m256 xDen;
    __m256 d1;
    __m256 d2;
    __m256 FutureValueX;
    __m256 NofXd1;
    __m256 NofXd2;
    __m256 NegNofXd1;
    __m256 NegNofXd2;    
    
    //new for sptprice/strike value
    __m256 DivTerm;
    float *DivFloat;
    __m256 MulHalf;

    xStockPrice = sptprice;
    xStrikePrice = strike;
    xRiskFreeRate = rate;
    xVolatility = volatility;

    xTime = time;
    xSqrtTime = _mm256_sqrt_ps(xTime);

    DivTerm = _mm256_div_ps(sptprice, strike);
    //DivFloat = (float*)DivTerm;
    //    logValues = log( sptprice / strike );
    logValues = _mm256_set_ps (log(((float*)&DivTerm)[7]),
			       log(((float*)&DivTerm)[6]),
			       log(((float*)&DivTerm)[5]),
			       log(((float*)&DivTerm)[4]), 
			       log(((float*)&DivTerm)[3]),
			       log(((float*)&DivTerm)[2]),
			       log(((float*)&DivTerm)[1]),
			       log(((float*)&DivTerm)[0])
			       );
    
    xLogTerm = logValues;
        
    
    xPowerTerm = _mm256_mul_ps(xVolatility , xVolatility);


    //xPowerTerm = xPowerTerm * 0.5;
    //MulHalf = _mm_set_ps1 ( 0.5);
    xPowerTerm = _mm256_mul_ps ( xPowerTerm, _mm256_set1_ps ( 0.5));
    
  
    //xD1 = xRiskFreeRate + xPowerTerm;
    xD1 = _mm256_add_ps ( xRiskFreeRate, xPowerTerm);

    //xD1 = xD1 * xTime;
    xD1 = _mm256_mul_ps( xD1, xTime);

    //xD1 = xD1 + xLogTerm;
    xD1 = _mm256_add_ps( xD1, xLogTerm);

    //xDen = xVolatility * xSqrtTime;
    xDen = _mm256_mul_ps(xVolatility, xSqrtTime);

    //xD1 = xD1 / xDen;
    xD1 = _mm256_div_ps(xD1, xDen);

    //xD2 = xD1 -  xDen;
    xD2 = _mm256_sub_ps(xD1, xDen);

    d1 = xD1;
    d2 = xD2;
    

    NofXd1 = CNDF( d1 );
    NofXd2 = CNDF( d2 );
    /*    int j;
    for ( j = 0; j < 4; j++){
      ((float*)&NofXd1)[j] = CNDF( ((float*)&d1)[j]);
      ((float*)&NofXd2)[j] = CNDF( ((float*)&d2)[j]);    
      }*/


    //FutureValueX = strike * ( exp( -(rate)*(time) ) );     
    __m256 rt = _mm256_mul_ps(rate, time);
    rt = _mm256_mul_ps(rt, _mm256_set1_ps(-1));
    float *rtfloat = (float*)&rt;
    __m256 expn = _mm256_set_ps (exp(rtfloat[7]),
				 exp(rtfloat[6]),
				 exp(rtfloat[5]),
				 exp(rtfloat[4]), 
				 exp(rtfloat[3]),
				 exp(rtfloat[2]),
				 exp(rtfloat[1]),
				 exp(rtfloat[0])
				 );

    FutureValueX = _mm256_mul_ps(strike, expn);
    
    

    
    /* if (otype == 0) {            
        OptionPrice = (sptprice * NofXd1) - (FutureValueX * NofXd2);
    } else { 
        NegNofXd1 = (1.0 - NofXd1);
        NegNofXd2 = (1.0 - NofXd2);
        OptionPrice = (FutureValueX * NegNofXd2) - (sptprice * NegNofXd1);
	}*/

    /* int i;
    for ( i = 0; i < 4; i++){
      if (((int*)&otype)[i] == 0) {  
	//printf("%ith otype is C\n", i);
	((float*)&OptionPrice)[i] = (((float*)&sptprice)[i] *((float*)&NofXd1)[i]) - (((float*)&FutureValueX)[i] * ((float*)&NofXd2)[i]);
      }
      else { 
	//printf("%ith otype is P\n", i);
	((float*)&NegNofXd1)[i] = (1.0 - ((float*)&NofXd1)[i]);
	((float*)&NegNofXd2)[i] = (1.0 - ((float*)&NofXd2)[i]);
	((float*)&OptionPrice)[i] = (((float*)&FutureValueX)[i] * ((float*)&NegNofXd2)[i]) - (((float*)&sptprice)[i] * ((float*)&NegNofXd1)[i]);
      }
      
      }*/
    
    __m256 ZeroValue;
    
    NegNofXd1 = _mm256_sub_ps(_mm256_cvtepi32_ps(otype), NofXd1);
    ZeroValue = _mm256_setzero_ps();
    ZeroValue = _mm256_sub_ps(ZeroValue, NegNofXd1);
    NegNofXd1 = _mm256_max_ps(ZeroValue, NegNofXd1);
    
    NegNofXd2 = _mm256_sub_ps(_mm256_cvtepi32_ps(otype), NofXd2);
    ZeroValue = _mm256_setzero_ps();
    ZeroValue = _mm256_sub_ps(ZeroValue, NegNofXd2);
    NegNofXd2 = _mm256_max_ps(ZeroValue, NegNofXd2);
    
    OptionPrice = _mm256_sub_ps(_mm256_mul_ps(sptprice, NegNofXd1),
			     _mm256_mul_ps(FutureValueX, NegNofXd2));
    //otype = _mm_cvtepi32_ps(otype);
    otype = _mm256_sub_epi32( _mm256_sub_epi32(_mm256_set1_epi32(1), otype),
			otype);
    OptionPrice = _mm256_mul_ps(OptionPrice, _mm256_cvtepi32_ps(otype));
    
    return OptionPrice;
}
 
//////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////
int bs_thread() {
  int i, j, l;
    //fptype price;
    fptype priceDelta;
    __m256 sptprice, strike, rate, volatility, otime, price;
    __m256i OptionType;

    for (j=0; j<NUM_RUNS; j++) { /* DO NOT swap the outer and the inner loop. This is for experiments. */
      for (i=0; i<numOptions; i+=8 ) {
	//	printf("%d th for loop\n", i);
	/* Calling main function to calculate option value based on 
	 * Black & Sholes's equation.
	 */

	// ([i],[i+1],[i+2],[i+3])
	/*	sptprice = _mm_set_ps(data[i+3].sptprice, data[i+2].sptprice, data[i+1].sptprice, data[i].sptprice);
	strike = _mm_set_ps(data[i+3].strike, data[i+2].strike, data[i+1].strike, data[i].strike);
	rate = _mm_set_ps(data[i+3].rate, data[i+2].rate, data[i+1].rate, data[i].rate);
	volatility = _mm_set_ps(data[i+3].volatility, data[i+2].volatility, data[i+1].volatility, data[i].volatility);
	otime = _mm_set_ps(data[i+3].otime, data[i+2].otime, data[i+1].otime, data[i].otime);
	OptionType =_mm_set_epi32(((data[i+3].OptionType == 'P') ? 1 : 0),
				  ((data[i+2].OptionType == 'P') ? 1 : 0), 
				  ((data[i+1].OptionType == 'P') ? 1 : 0), 
				  ((data[i].OptionType == 'P') ? 1 : 0)
				  );*/ 
	/*
	for ( l = 0; l < 4; l++){
	  printf(" bs_thread OptionType __m128, %dth: %d\n", l, ((int*)&OptionType)[l]);
	  printf(" data[%d].OptionType is %c\n", i+l, data[i+l].OptionType);
	  }*/
	sptprice = _mm256_loadu_ps(&data.sptprice[i]);
	strike = _mm256_loadu_ps(&data.strike[i]);
	rate = _mm256_loadu_ps(&data.rate[i]);
	volatility = _mm256_loadu_ps(&data.volatility[i]);
	otime = _mm256_loadu_ps(&data.otime[i]);
	OptionType = _mm256_set_epi32(((data.OptionType[i+7] == 'P') ? 1 : 0),
				      ((data.OptionType[i+6] == 'P') ? 1 : 0),
				      ((data.OptionType[i+5] == 'P') ? 1 : 0),
				      ((data.OptionType[i+4] == 'P') ? 1 : 0),
				      ((data.OptionType[i+3] == 'P') ? 1 : 0),
				      ((data.OptionType[i+2] == 'P') ? 1 : 0), 
				      ((data.OptionType[i+1] == 'P') ? 1 : 0), 
				      ((data.OptionType[i] == 'P') ? 1 : 0)
				      );

	price = BlkSchlsEqEuroNoDiv( sptprice, 
				     strike,
				     rate,
				     volatility, 
				     otime, 
				     OptionType 
				     );
	
	//prices[i] = price;
	_mm256_storeu_ps(&prices[i], price); 
      }
    }
    
    return 0;
}

int main (int argc, char **argv)
{
  FILE *file;
  int i;
    int loopnum;
    int rv;
    char *inputFile = NULL;
    char *outputFile = NULL;
    struct timeval exec_begin, exec_end, roi_begin, roi_end;
    
    gettimeofday(&exec_begin, 0);
    printf("Blackscholes from PARSEC Benchmark Suite\n");
    fflush(NULL);
    
    if (argc < 3 || argc > 4) {
      printf("Usage:\n\t%s <inputFile> <outputFile> <nthreads>\n", argv[0]);
      printf("'nthreads' can be omitted (default: 1)\n");
      exit(1);
    }
    
    inputFile = argv[1];
    outputFile = argv[2];
    
    nThreads = 1;
    if (argc == 4)
      nThreads = atoi(argv[3]);
    
    //Read input data from file
    file = fopen(inputFile, "r");
    if(file == NULL) {
      printf("ERROR: Unable to open file `%s'.\n", inputFile);
      exit(1);
    }
    rv = fscanf(file, "%i", &numOptions);
    if(rv != 1) {
      printf("ERROR: Unable to read from file `%s'.\n", inputFile);
      fclose(file);
      exit(1);
    }

	/*printf("INFO: sizeof(OptionData): %ld numOptions: %ld sizeof(data): %ld sizeof(prices): %ld sizeof(fptype): %ld\n", 
			sizeof(OptionData), numOptions, numOptions * sizeof(OptionData), numOptions*sizeof(fptype), sizeof(fptype));*/
    // alloc spaces for the option data

    //data = (OptionData*)malloc(numOptions*sizeof(OptionData));  //MALOC
    data.sptprice = (fptype*)malloc(numOptions*sizeof(fptype));
    data.strike = (fptype*)malloc(numOptions*sizeof(fptype));
    data.rate = (fptype*)malloc(numOptions*sizeof(fptype));
    data.divq = (fptype*)malloc(numOptions*sizeof(fptype));
    data.volatility = (fptype*)malloc(numOptions*sizeof(fptype));
    data.otime = (fptype*)malloc(numOptions*sizeof(fptype));
    data.OptionType = (char*)malloc(numOptions*sizeof(char));
    data.divs = (fptype*)malloc(numOptions*sizeof(fptype));
    data.DGrefval = (fptype*)malloc(numOptions*sizeof(fptype));

    // data = (OptionData*)aligned_alloc(16, numOptions*sizeof(OptionData));
    prices = (fptype*)malloc(numOptions*sizeof(fptype));
    for ( loopnum = 0; loopnum < numOptions; ++ loopnum )
    {
        rv = fscanf(file, "%f %f %f %f %f %f %c %f %f", 
							&data.sptprice[loopnum], 
							&data.strike[loopnum], 
							&data.rate[loopnum],
							&data.divq[loopnum],
							&data.volatility[loopnum],
							&data.otime[loopnum],
							&data.OptionType[loopnum],
							&data.divs[loopnum],
							&data.DGrefval[loopnum]);
        if(rv != 9) {
          printf("ERROR: Unable to read from file `%s'.\n", inputFile);
          fclose(file);
          exit(1);
        }
    }
    rv = fclose(file);
    if(rv != 0) {
      printf("ERROR: Unable to close file `%s'.\n", inputFile);
      exit(1);
    }

    printf("Num of Options: %d\n", numOptions);
    printf("Num of Runs: %d\n", NUM_RUNS);

	/* working set: data + prices */
	rv = numOptions * sizeof(OptionData) + numOptions * sizeof(fptype);
	printf("Size of working set: %d\n", rv); 

	/* Region of Interest begins */
	gettimeofday(&roi_begin, 0);

    bs_thread();
	
	/* Region of Interest ends */
	gettimeofday(&roi_end, 0);

    //Write prices to output file
    file = fopen(outputFile, "w");
    if(file == NULL) {
      printf("ERROR: Unable to open file `%s'.\n", outputFile);
      exit(1);
    }
    rv = fprintf(file, "%i\n", numOptions);
    if(rv < 0) {
      printf("ERROR: Unable to write to file `%s'.\n", outputFile);
      fclose(file);
      exit(1);
    }
    for(i=0; i<numOptions; i++) {
      rv = fprintf(file, "%.3f\n", prices[i]);
      if(rv < 0) {
        printf("ERROR: Unable to write to file `%s'.\n", outputFile);
        fclose(file);
        exit(1);
      }
    }
    rv = fclose(file);
    if(rv != 0) {
      printf("ERROR: Unable to close file `%s'.\n", outputFile);
      exit(1);
    }

    free(data.sptprice);
    free(data.strike); 
    free(data.rate);
    free(data.divq);
    free(data.volatility);
    free(data.otime);
    free(data.OptionType);
    free(data.divs);
    free(data.DGrefval);


    //free(data);
    free(prices);

	gettimeofday(&exec_end, 0);

	printf("Time(ROI): %.3f\n", TIME_DIFF(roi_begin, roi_end));
	printf("Time(total): %.3f\n", TIME_DIFF(exec_begin, exec_end));

    return 0;
}

