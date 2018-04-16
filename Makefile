all : bs_origin

bs_origin : bs_origin.c
	gcc -o main bs_origin.c -lm
