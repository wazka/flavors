NVCC=nvcc
SRC=src
BIN=bin

CC=35
#CC=61

NVCC_FLAGS=-rdc=true -gencode arch=compute_$(CC),code=sm_$(CC) -std=c++11 -I lib/cuda-api-wrappers/api/ -I lib/cuda-api-wrappers/ -I src/

LIB=$(BIN)/tmp/device_properties.o
FLAVORS=$(BIN)/tmp/configuration.o $(BIN)/tmp/keys.o $(BIN)/tmp/masks.o $(BIN)/tmp/tree.o $(BIN)/tmp/utils.o
SAMPLE=$(BIN)/basicKeysSample $(BIN)/basicMasksSample

all: lib flavors

lib: bin $(LIB)
flavors: bin $(FLAVORS) $(BIN)/flavors.a
sample: flavors $(SAMPLE)

# flavors library objects
$(BIN)/flavors.a: $(LIB) $(FLAVORS) 
	ar rcs $(BIN)/flavors.a $(LIB) $(FLAVORS)

$(BIN)/tmp/configuration.o: $(SRC)/configuration.cu
	$(NVCC) $(NVCC_FLAGS) -c $(SRC)/configuration.cu -o $(BIN)/tmp/configuration.o

$(BIN)/tmp/keys.o: $(SRC)/keys.cu
	$(NVCC) $(NVCC_FLAGS) -c $(SRC)/keys.cu -o $(BIN)/tmp/keys.o

$(BIN)/tmp/masks.o: $(SRC)/masks.cu
	$(NVCC) $(NVCC_FLAGS) -c $(SRC)/masks.cu -o $(BIN)/tmp/masks.o

$(BIN)/tmp/tree.o: $(SRC)/tree.cu
	$(NVCC) $(NVCC_FLAGS) -c $(SRC)/tree.cu -o $(BIN)/tmp/tree.o

$(BIN)/tmp/utils.o: $(SRC)/utils.cpp
	$(NVCC) $(NVCC_FLAGS) -c $(SRC)/utils.cpp -o $(BIN)/tmp/utils.o

# cuda-api-wrappers objects
$(BIN)/tmp/device_properties.o: lib/cuda-api-wrappers/api/device_properties.cpp
	$(NVCC) $(NVCC_FLAGS) -c lib/cuda-api-wrappers/api/device_properties.cpp -o $(BIN)/tmp/device_properties.o

# samples objects
$(BIN)/basicKeysSample: $(BIN)/tmp/basicKeysSample.o
	$(NVCC) $(NVCC_FLAGS) -o $(BIN)/basicKeysSample $(BIN)/tmp/basicKeysSample.o $(BIN)/flavors.a

$(BIN)/tmp/basicKeysSample.o: sample/basicKeysSample.cpp
	$(NVCC) $(NVCC_FLAGS) -c sample/basicKeysSample.cpp -o $(BIN)/tmp/basicKeysSample.o

$(BIN)/basicMasksSample: $(BIN)/tmp/basicMasksSample.o
	$(NVCC) $(NVCC_FLAGS) -o $(BIN)/basicMasksSample $(BIN)/tmp/basicMasksSample.o $(BIN)/flavors.a

$(BIN)/tmp/basicMasksSample.o: sample/basicMasksSample.cpp
	$(NVCC) $(NVCC_FLAGS) -c sample/basicMasksSample.cpp -o $(BIN)/tmp/basicMasksSample.o

bin:
	mkdir -p $(BIN)
	mkdir -p $(BIN)/tmp

clear: clean
	rm -f $(BIN)/*.o
	rm -r -f $(BIN)

clean:	
	rm -f $(BIN)/tmp/*.o
	rm -r -f $(BIN)/tmp

.PHONY: all clear clean sample

.PHONY: all clear clean


