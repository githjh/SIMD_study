all : bs_origin

bs_origin : bs_origin.c ; gcc main -o bs_origin.c -lm
