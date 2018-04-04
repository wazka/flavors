SHELL=powershell.exe
NVCC=nvcc
SRC=flavors/
BENCH_SRC=benchmarks/
SAMPLE_SRC=samples/
BIN_DIR=./bin
LIB_DIR=lib
INCLUDES=-I $(LIB_DIR)/cuda-api-wrappers/api/ -I $(LIB_DIR)/cuda-api-wrappers/ -I $(LIB_DIR)/json -I $(SRC)/ -I benchmark/

NVCC_FLAGS=-rdc=true -gencode arch=compute_50,code=sm_50 -std=c++11 -O3 $(INCLUDES)

BIN=$(BIN_DIR) $(BIN_DIR)/tmp
LIB=$(BIN_DIR)/tmp/device_properties.o
FLAVORS=$(BIN_DIR)/tmp/configuration.o $(BIN_DIR)/tmp/keys.o $(BIN_DIR)/tmp/masks.o $(BIN_DIR)/tmp/tree.o $(BIN_DIR)/tmp/utils.o $(BIN_DIR)/tmp/dataInfo.o
BENCHMARKS=$(BIN_DIR)/tmp/benchmark.o $(BIN_DIR)/tmp/dictionary.o $(BIN_DIR)/tmp/words.o $(BIN_DIR)/tmp/ip.o  $(BIN_DIR)/tmp/runMain.o
SAMPLE=$(BIN_DIR)/tmp/keysSample.o $(BIN_DIR)/tmp/longKeysSample.o

all: flavors

lib: $(BIN) $(LIB)
flavors: $(BIN) $(FLAVORS) $(BIN_DIR)/flavors.a lib
benchmarks: $(BIN) flavors $(BENCHMARKS) $(BIN_DIR)/flavors-benchmarks
samples: $(BIN) $(FLAVORS) $(SAMPLE) $(BIN_DIR)/keys-sample $(BIN_DIR)/long-keys-sample

# flavors library objects
$(BIN_DIR)/flavors.a: $(LIB) $(FLAVORS) $(SRC)/containers.h
	ar rcs $(BIN_DIR)/flavors.a $(LIB) $(FLAVORS)

$(BIN_DIR)/tmp/configuration.o: $(SRC)/configuration.cu $(SRC)/configuration.h
	$(NVCC) $(NVCC_FLAGS) -c $(SRC)/configuration.cu -o $(BIN_DIR)/tmp/configuration.o

$(BIN_DIR)/tmp/keys.o: $(SRC)/keys.cu $(SRC)/keys.h
	$(NVCC) $(NVCC_FLAGS) -c $(SRC)/keys.cu -o $(BIN_DIR)/tmp/keys.o

$(BIN_DIR)/tmp/masks.o: $(SRC)/masks.cu $(SRC)/masks.h
	$(NVCC) $(NVCC_FLAGS) -c $(SRC)/masks.cu -o $(BIN_DIR)/tmp/masks.o

$(BIN_DIR)/tmp/tree.o: $(SRC)/tree.cu $(SRC)/tree.h
	$(NVCC) $(NVCC_FLAGS) -c $(SRC)/tree.cu -o $(BIN_DIR)/tmp/tree.o

$(BIN_DIR)/tmp/utils.o: $(SRC)/utils.cpp $(SRC)/utils.h
	$(NVCC) $(NVCC_FLAGS) -c $(SRC)/utils.cpp -o $(BIN_DIR)/tmp/utils.o

$(BIN_DIR)/tmp/dataInfo.o: $(SRC)/dataInfo.cu $(SRC)/dataInfo.h
	$(NVCC) $(NVCC_FLAGS) -c $(SRC)/dataInfo.cu -o $(BIN_DIR)/tmp/dataInfo.o

# cuda-api-wrappers objects
$(BIN_DIR)/tmp/device_properties.o: $(LIB_DIR)/cuda-api-wrappers/api/device_properties.cpp
	$(NVCC) $(NVCC_FLAGS) -c $(LIB_DIR)/cuda-api-wrappers/api/device_properties.cpp -o $(BIN_DIR)/tmp/device_properties.o

# benchmarks objects
$(BIN_DIR)/tmp/benchmark.o: $(BENCH_SRC)/benchmark.cpp $(BENCH_SRC)/benchmark.h
	$(NVCC) $(NVCC_FLAGS) -c $(BENCH_SRC)/benchmark.cpp -o $(BIN_DIR)/tmp/benchmark.o

$(BIN_DIR)/tmp/dictionary.o: $(BENCH_SRC)/dictionary.cpp $(BENCH_SRC)/dictionary.h
	$(NVCC) $(NVCC_FLAGS) -c $(BENCH_SRC)/dictionary.cpp -o $(BIN_DIR)/tmp/dictionary.o

$(BIN_DIR)/tmp/words.o: $(BENCH_SRC)/words.cpp $(BENCH_SRC)/words.h
	$(NVCC) $(NVCC_FLAGS) -c $(BENCH_SRC)/words.cpp -o $(BIN_DIR)/tmp/words.o

$(BIN_DIR)/tmp/ip.o: $(BENCH_SRC)/ip.cpp $(BENCH_SRC)/ip.h
	$(NVCC) $(NVCC_FLAGS) -c $(BENCH_SRC)/ip.cpp -o $(BIN_DIR)/tmp/ip.o

$(BIN_DIR)/tmp/runMain.o: $(BENCH_SRC)/runMain.cpp
	$(NVCC) $(NVCC_FLAGS) -c $(BENCH_SRC)/runMain.cpp -o $(BIN_DIR)/tmp/runMain.o

$(BIN_DIR)/flavors-benchmarks: $(BENCH_SRC)/hostBenchmark.h $(BENCH_SRC)/randomBenchmark.h $(BENCHMARKS) $(BIN_DIR)/flavors.a
	$(NVCC) $(NVCC_FLAGS) -o $(BIN_DIR)/flavors-benchmarks $(BIN_DIR)/flavors.a $(BENCHMARKS)

# sample
$(BIN_DIR)/tmp/keysSample.o: $(SAMPLE_SRC)/keysSample.cpp
	$(NVCC) $(NVCC_FLAGS) -c $(SAMPLE_SRC)/keysSample.cpp -o $(BIN_DIR)/tmp/keysSample.o

$(BIN_DIR)/keys-sample: $(SAMPLE) $(FLAVORS)
	$(NVCC) $(NVCC_FLAGS) -o $(BIN_DIR)/keys-sample $(FLAVORS) $(BIN_DIR)/tmp/keysSample.o

$(BIN_DIR)/tmp/longKeysSample.o:  $(SAMPLE_SRC)/longKeysSample.cpp
	$(NVCC) $(NVCC_FLAGS) -c $(SAMPLE_SRC)/longKeysSample.cpp -o $(BIN_DIR)/tmp/longKeysSample.o

$(BIN_DIR)/long-keys-sample: $(SAMPLE) $(FLAVORS)
	$(NVCC) $(NVCC_FLAGS) -o $(BIN_DIR)/long-keys-sample $(FLAVORS) $(BIN_DIR)/tmp/longKeysSample.o

$(BIN_DIR):
	mkdir -p "$(BIN_DIR)"

$(BIN_DIR)/tmp: 
	mkdir -p "$(BIN_DIR)/tmp"

clear: clean
	rm -Force "$(BIN_DIR)/*.o"
	rm -r -Force "$(BIN_DIR)"

clean:	
	rm -Force "$(BIN_DIR)/tmp/*.o"
	rm -r -Force "$(BIN_DIR)/tmp"

.PHONY: all clear clean