NVCC=nvcc
SRC=flavors/src
BENCH_SRC=benchmark/flavors-benchmark/src
BIN=bin


NVCC_FLAGS=-rdc=true -gencode arch=compute_61,code=sm_61 -std=c++11 -O3  -I lib/cuda-api-wrappers/api/ -I lib/cuda-api-wrappers/ -I lib/json -I $(SRC)/ -I benchmark/flavors-benchmark/src

LIB=$(BIN)/tmp/device_properties.o
FLAVORS=$(BIN)/tmp/configuration.o $(BIN)/tmp/keys.o $(BIN)/tmp/masks.o $(BIN)/tmp/tree.o $(BIN)/tmp/utils.o $(BIN)/tmp/dataInfo.o
BENCHMARKS=$(BIN)/tmp/benchmark.o $(BIN)/tmp/dictionary.o $(BIN)/tmp/words.o $(BIN)/tmp/runMain.o

all: lib flavors benchmarks

lib: bin $(LIB)
flavors: bin $(FLAVORS) $(BIN)/flavors.a
benchmarks: bin flavors $(BENCHMARKS) $(BIN)/flavors-benchmarks

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

$(BIN)/tmp/dataInfo.o: $(SRC)/dataInfo.cu
	$(NVCC) $(NVCC_FLAGS) -c $(SRC)/dataInfo.cu -o $(BIN)/tmp/dataInfo.o

# cuda-api-wrappers objects
$(BIN)/tmp/device_properties.o: lib/cuda-api-wrappers/api/device_properties.cpp
	$(NVCC) $(NVCC_FLAGS) -c lib/cuda-api-wrappers/api/device_properties.cpp -o $(BIN)/tmp/device_properties.o

# benchmarks objects
$(BIN)/tmp/benchmark.o: $(BENCH_SRC)/benchmark.cpp
	$(NVCC) $(NVCC_FLAGS) -c $(BENCH_SRC)/benchmark.cpp -o $(BIN)/tmp/benchmark.o

$(BIN)/tmp/dictionary.o: $(BENCH_SRC)/dictionary.cpp
	$(NVCC) $(NVCC_FLAGS) -c $(BENCH_SRC)/dictionary.cpp -o $(BIN)/tmp/dictionary.o

$(BIN)/tmp/words.o: $(BENCH_SRC)/words.cpp
	$(NVCC) $(NVCC_FLAGS) -c $(BENCH_SRC)/words.cpp -o $(BIN)/tmp/words.o

$(BIN)/tmp/runMain.o: benchmark/benchmark-run/src/runMain.cpp
	$(NVCC) $(NVCC_FLAGS) -c benchmark/benchmark-run/src/runMain.cpp -o $(BIN)/tmp/runMain.o

$(BIN)/flavors-benchmarks:
	$(NVCC) $(NVCC_FLAGS) -o $(BIN)/flavors-benchmarks $(BIN)/flavors.a $(BENCHMARKS)

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


