all : bs_origin bs_origin_ori

bs_origin : bs_origin.c
	gcc -o main bs_origin.c -lm
bs_origin_ori : bs_origin_ori.c
	gcc -o main_ori bs_origin_ori.c -lm
