CXX = g++
CXXFLAGS = -O2 -Wall -std=c++11 `pkg-config --cflags opencv` `pkg-config --libs opencv`
LDFLAGS = -lm
all: imageProcess

run:
	./imageProcess

clean:
	$(RM) imageProcess
