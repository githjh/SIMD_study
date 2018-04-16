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
#include <emmintrin.h>

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

OptionData *data;
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

fptype CNDF ( fptype InputX ) 
{
    int sign;

    fptype f[4];

    fptype OutputX;
    fptype xInput;
    fptype xNPrimeofX;
    fptype expValues;
    fptype xK2;
    fptype xK2_2, xK2_3;
    fptype xK2_4, xK2_5;
    fptype xLocal, xLocal_1;
    fptype xLocal_2, xLocal_3;

    // Check for negative value of InputX
    if (InputX < 0.0) {
        InputX = -InputX;
        sign = 1;
    } else 
        sign = 0;

    xInput = InputX;
 
    // Compute NPrimeX term common to both four & six decimal accuracy calcs
    expValues = exp(-0.5f * InputX * InputX);
    xNPrimeofX = expValues;
    
    xNPrimeofX = xNPrimeofX * inv_sqrt_2xPI;
    xK2 = 0.2316419 * xInput;  
    xK2 = 1.0 + xK2;
    xK2 = 1.0 / xK2;
    
    xK2_2 = xK2 * xK2;
    xK2_3 = xK2_2 * xK2;
    xK2_4 = xK2_3 * xK2;
    xK2_5 = xK2_4 * xK2;
     
    xLocal_1 = xK2 * 0.319381530;
    xLocal_2 = xK2_2 * (-0.356563782);
    xLocal_3 = xK2_3 * 1.781477937;

    xLocal_2 = xLocal_2 + xLocal_3;
    xLocal_3 = xK2_4 * (-1.821255978);
    xLocal_2 = xLocal_2 + xLocal_3;
    xLocal_3 = xK2_5 * 1.330274429;
    xLocal_2 = xLocal_2 + xLocal_3;

    xLocal_1 = xLocal_2 + xLocal_1;
    xLocal   = xLocal_1 * xNPrimeofX;
    xLocal   = 1.0 - xLocal;

    OutputX  = xLocal;
    
    if (sign) {
        OutputX = 1.0 - OutputX;
    }
    
    return OutputX;
} 

//////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////

fptype BlkSchlsEqEuroNoDiv_SIMD( OptionData * data_list )
{
    fptype OptionPrice;

    // local private working variables for the calculation
    //fptype xStockPrice;
    //fptype xStrikePrice;
    //fptype xRiskFreeRate;
    //fptype xVolatility;
    //fptype xTime;
    //fptype xSqrtTime;

//    fptype logValues;
    fptype xLogTerm;
    fptype xD1; 
    fptype xD2;
    fptype xPowerTerm;
    fptype xDen;
    fptype d1;
    fptype d2;
    fptype FutureValueX;
    fptype NofXd1;
    fptype NofXd2;
    fptype NegNofXd1;
    fptype NegNofXd2;    
    
    fptype xStockPrice [4] = {data_list[0].sptprice, data_list[1].sptprice, 
			      data_list[2].sptprice, data_list[3].sptprice};
    fptype xStrikePrice [4] = {data_list[0].strike, data_list[1].strike, 
			      data_list[2].strike, data_list[3].strike};
    fptype xRiskFreeRate [4] = {data_list[0].rate, data_list[1].rate, 
			      data_list[2].rate, data_list[3].rate};
    fptype xVolatility [4] = {data_list[0].volatility, data_list[1].volatility, 
			      data_list[2].volatility, data_list[3].volatility};
    fptype xTime [4] = {data_list[0].otime, data_list[1].otime,
			data_list[2].otime, data_list[3].otime};
    fptype xSqrtTime [4] = {sqrt(xTime[0]), sqrt(xTime[1]), sqrt(xTime[2]),sqrt(xTime[3])};

    fptype logValues [4] = {log (xStockPrice[0]/xStrikePrice[0]),log (xStockPrice[1]/xStrikePrice[1]),
                           log (xStockPrice[2]/xStrikePrice[2]),log (xStockPrice[3]/xStrikePrice[3])};
#if 0
    xStockPrice = sptprice;
    xStrikePrice = strike;
    xRiskFreeRate = rate;
    xVolatility = volatility;

    xTime = time;
    xSqrtTime = sqrt(xTime);

    logValues = log( sptprice / strike );
        
    xLogTerm = logValues;
        
    
    xPowerTerm = xVolatility * xVolatility;
    xPowerTerm = xPowerTerm * 0.5;
        
    xD1 = xRiskFreeRate + xPowerTerm;
    xD1 = xD1 * xTime;
    xD1 = xD1 + xLogTerm;

    xDen = xVolatility * xSqrtTime;
    xD1 = xD1 / xDen;
    xD2 = xD1 -  xDen;

    d1 = xD1;
    d2 = xD2;
    
    NofXd1 = CNDF( d1 );
    NofXd2 = CNDF( d2 );

    FutureValueX = strike * ( exp( -(rate)*(time) ) );        
    
    if (otype == 0) {            
        OptionPrice = (sptprice * NofXd1) - (FutureValueX * NofXd2);
    } else { 
        NegNofXd1 = (1.0 - NofXd1);
        NegNofXd2 = (1.0 - NofXd2);
        OptionPrice = (FutureValueX * NegNofXd2) - (sptprice * NegNofXd1);
    }
    
    return OptionPrice;
#endif
    return 0;
}

fptype BlkSchlsEqEuroNoDiv( fptype sptprice,
                            fptype strike, fptype rate, fptype volatility,
                            fptype time, int otype )
{
    fptype OptionPrice;

    // local private working variables for the calculation
    fptype xStockPrice;
    fptype xStrikePrice;
    fptype xRiskFreeRate;
    fptype xVolatility;
    fptype xTime;
    fptype xSqrtTime;

    fptype logValues;
    fptype xLogTerm;
    fptype xD1; 
    fptype xD2;
    fptype xPowerTerm;
    fptype xDen;
    fptype d1;
    fptype d2;
    fptype FutureValueX;
    fptype NofXd1;
    fptype NofXd2;
    fptype NegNofXd1;
    fptype NegNofXd2;    
    
    xStockPrice = sptprice;
    xStrikePrice = strike;
    xRiskFreeRate = rate;
    xVolatility = volatility;

    xTime = time;
    xSqrtTime = sqrt(xTime);

    logValues = log( sptprice / strike );
        
    xLogTerm = logValues;
        
    
    xPowerTerm = xVolatility * xVolatility;
    xPowerTerm = xPowerTerm * 0.5;
        
    xD1 = xRiskFreeRate + xPowerTerm;
    xD1 = xD1 * xTime;
    xD1 = xD1 + xLogTerm;

    xDen = xVolatility * xSqrtTime;
    xD1 = xD1 / xDen;
    xD2 = xD1 -  xDen;

    d1 = xD1;
    d2 = xD2;
    
    NofXd1 = CNDF( d1 );
    NofXd2 = CNDF( d2 );

    FutureValueX = strike * ( exp( -(rate)*(time) ) );        
    
    if (otype == 0) {            
        OptionPrice = (sptprice * NofXd1) - (FutureValueX * NofXd2);
    } else { 
        NegNofXd1 = (1.0 - NofXd1);
        NegNofXd2 = (1.0 - NofXd2);
        OptionPrice = (FutureValueX * NegNofXd2) - (sptprice * NegNofXd1);
    }
    
    return OptionPrice;
}

//////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////
int bs_thread() {
    int i, j;
    fptype price;
    fptype priceDelta;

    for (j=0; j<NUM_RUNS; j++) { /* DO NOT swap the outer and the inner loop. This is for experiments. */
        for (i=0; i<numOptions; i = i + 8) {
            /* Calling main function to calculate option value based on 
             * Black & Sholes's equation.
             */
            price = BlkSchlsEqEuroNoDiv( data[i].sptprice, 
					 data[i].strike,
                                         data[i].rate,
					 data[i].volatility, 
					 data[i].otime, 
                                         (data[i].OptionType == 'P') ? 1 : 0 
					 );

            prices[i] = price;
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
    data = (OptionData*)malloc(numOptions*sizeof(OptionData));
    prices = (fptype*)malloc(numOptions*sizeof(fptype));
    for ( loopnum = 0; loopnum < numOptions; ++ loopnum )
    {
        rv = fscanf(file, "%f %f %f %f %f %f %c %f %f", 
							&data[loopnum].sptprice, 
							&data[loopnum].strike, 
							&data[loopnum].rate,
							&data[loopnum].divq,
							&data[loopnum].volatility,
							&data[loopnum].otime,
							&data[loopnum].OptionType,
							&data[loopnum].divs,
							&data[loopnum].DGrefval);
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

    free(data);
    free(prices);

	gettimeofday(&exec_end, 0);

	printf("Time(ROI): %.3f\n", TIME_DIFF(roi_begin, roi_end));
	printf("Time(total): %.3f\n", TIME_DIFF(exec_begin, exec_end));

    return 0;
}

