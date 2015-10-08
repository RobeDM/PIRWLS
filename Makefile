CC=gcc

OPTFLAGS = -O3

USE_MKL=0

ifeq ($(USE_MKL),1)
	CFLAGS= -USE_MKL
	LIBS = -lmkl_core -lmkl_sequential -lmkl_intel_lp64 -lm
	INCLUDEPATH = -I/opt/intel/composerxe-2013.1.106/mkl/include/
	LIBRARYPATH = -L/opt/intel/composerxe-2013.1.106/mkl/lib/intel64/
else
	LIBS = -lcblas
  INCLUDEPATH = -I/usr/local/atlas/include/
  LIBRARYPATH = -L/usr/local/atlas/lib/
endif

all: PIRWLS-train PIRWLS-predict

PIRWLS-train: PIRWLS-train.c
	$(CC) $(OPTFLAGS) $(CFLAGS) $(INCLUDEPATH) $(LIBRARYPATH) -o PIRWLS-train PIRWLS-train.c $(LIBS)

PIRWLS-predict: PIRWLS-predict.c
	$(CC) $(OPTFLAGS) $(CFLAGS) $(INCLUDEPATH) $(LIBRARYPATH) -o PIRWLS-predict PIRWLS-predict.c $(LIBS)

clean:
	rm -f PIRWLS-{train,predict}
