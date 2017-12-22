

INCLUDES = -Imnist_reader -IVAE -I.
CPP_FILES = VAE/EncDecder.cpp \
			VAE/vae.cpp \
			trainer.hpp \
			trainer.cpp \
			main.cpp
LDFLAGS =
CFLAGS =  -std=gnu++11 -g3 -Wall -Wextra
FLAGS = $(INCLUDES) $(CFLAGS) $(LDFLAGS)
CC = g++
all: VAECPP


VAECPP: main.cpp
	$(CC) $(FLAGS) $(CPP_FILES) -o VAECPP

run: VAECPP
	./VAECPP
debug: VAECPP
	ddd VAECPP
clean:
	rm VAECPP