SHELL=powershell.exe
NVCC=nvcc
SRC=flavors/
BENCH_SRC=benchmarks/
SAMPLE_SRC=samples/
TEST_SRC=test/
BIN_DIR=./bin
LIB_DIR=lib
TEST_SRC=test/
INCLUDES=-I $(LIB_DIR)/cuda-api-wrappers/api/ -I $(LIB_DIR)/cuda-api-wrappers/ -I $(LIB_DIR)/json -I $(LIB_DIR)/catch -I "C:\Program Files\boost\boost_1_66_0" -I $(SRC)/ -I benchmark/

NVCC_FLAGS=-rdc=true -gencode arch=compute_50,code=sm_50 -std=c++11 -O3 $(INCLUDES)

BIN=$(BIN_DIR) $(BIN_DIR)/tmp
FLAVORS=$(BIN_DIR)/tmp/configuration.o $(BIN_DIR)/tmp/keys.o $(BIN_DIR)/tmp/masks.o $(BIN_DIR)/tmp/tree.o $(BIN_DIR)/tmp/compressedTree.o $(BIN_DIR)/tmp/utils.o $(BIN_DIR)/tmp/dataInfo.o
BENCHMARKS=$(BIN_DIR)/tmp/benchmark.o $(BIN_DIR)/tmp/dictionary.o $(BIN_DIR)/tmp/words.o $(BIN_DIR)/tmp/ip.o  $(BIN_DIR)/tmp/runMain.o
SAMPLE=$(BIN_DIR)/tmp/keysSample.o $(BIN_DIR)/tmp/longKeysSample.o
TEST=$(BIN_DIR)/tmp/runTests.o $(BIN_DIR)/tmp/testConfiguration.o $(BIN_DIR)/tmp/testKeys.o $(BIN_DIR)/tmp/testMasks.o $(BIN_DIR)/tmp/testTree.o $(BIN_DIR)/tmp/testLoad.o $(BIN_DIR)/tmp/helpers.o

all: flavors

flavors: $(BIN) $(FLAVORS) $(BIN_DIR)/flavors.a
benchmarks: $(BIN) $(FLAVORS) $(BENCHMARKS) $(BIN_DIR)/flavors-benchmarks
samples: $(BIN) $(FLAVORS) $(SAMPLE) $(BIN_DIR)/keys-sample $(BIN_DIR)/long-keys-sample
tests: $(BIN) $(FLAVORS) $(TEST) $(BIN_DIR)/flavors-tests

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

$(BIN_DIR)/tmp/compressedTree.o: $(SRC)/compressedTree.cu $(SRC)/compressedTree.h
	$(NVCC) $(NVCC_FLAGS) -c $(SRC)/compressedTree.cu -o $(BIN_DIR)/tmp/compressedTree.o

$(BIN_DIR)/tmp/utils.o: $(SRC)/utils.cpp $(SRC)/utils.h
	$(NVCC) $(NVCC_FLAGS) -c $(SRC)/utils.cpp -o $(BIN_DIR)/tmp/utils.o

$(BIN_DIR)/tmp/dataInfo.o: $(SRC)/dataInfo.cu $(SRC)/dataInfo.h
	$(NVCC) $(NVCC_FLAGS) -c $(SRC)/dataInfo.cu -o $(BIN_DIR)/tmp/dataInfo.o

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

$(BIN_DIR)/flavors-benchmarks: $(BENCH_SRC)/hostBenchmark.h $(BENCH_SRC)/randomBenchmark.h $(BENCHMARKS) $(FLAVORS)
	$(NVCC) $(NVCC_FLAGS) -o $(BIN_DIR)/flavors-benchmarks $(FLAVORS) $(BENCHMARKS)

# sample
$(BIN_DIR)/tmp/keysSample.o: $(SAMPLE_SRC)/keysSample.cpp
	$(NVCC) $(NVCC_FLAGS) -c $(SAMPLE_SRC)/keysSample.cpp -o $(BIN_DIR)/tmp/keysSample.o

$(BIN_DIR)/keys-sample: $(SAMPLE) $(FLAVORS)
	$(NVCC) $(NVCC_FLAGS) -o $(BIN_DIR)/keys-sample $(FLAVORS) $(BIN_DIR)/tmp/keysSample.o

$(BIN_DIR)/tmp/longKeysSample.o:  $(SAMPLE_SRC)/longKeysSample.cpp
	$(NVCC) $(NVCC_FLAGS) -c $(SAMPLE_SRC)/longKeysSample.cpp -o $(BIN_DIR)/tmp/longKeysSample.o

$(BIN_DIR)/long-keys-sample: $(SAMPLE) $(FLAVORS)
	$(NVCC) $(NVCC_FLAGS) -o $(BIN_DIR)/long-keys-sample $(FLAVORS) $(BIN_DIR)/tmp/longKeysSample.o

#tests
$(BIN_DIR)/tmp/testConfiguration.o:  $(TEST_SRC)/testConfiguration.cpp
	$(NVCC) $(NVCC_FLAGS) -c $(TEST_SRC)/testConfiguration.cpp -o $(BIN_DIR)/tmp/testConfiguration.o

$(BIN_DIR)/tmp/testKeys.o:  $(TEST_SRC)/testKeys.cpp
	$(NVCC) $(NVCC_FLAGS) -c $(TEST_SRC)/testKeys.cpp -o $(BIN_DIR)/tmp/testKeys.o

$(BIN_DIR)/tmp/testMasks.o:  $(TEST_SRC)/testMasks.cpp
	$(NVCC) $(NVCC_FLAGS) -c $(TEST_SRC)/testMasks.cpp -o $(BIN_DIR)/tmp/testMasks.o

$(BIN_DIR)/tmp/testTree.o:  $(TEST_SRC)/testTree.cpp
	$(NVCC) $(NVCC_FLAGS) -c $(TEST_SRC)/testTree.cpp -o $(BIN_DIR)/tmp/testTree.o

$(BIN_DIR)/tmp/testLoad.o:  $(TEST_SRC)/testLoad.cpp
	$(NVCC) $(NVCC_FLAGS) -c $(TEST_SRC)/testLoad.cpp -o $(BIN_DIR)/tmp/testLoad.o

$(BIN_DIR)/tmp/helpers.o:  $(TEST_SRC)/helpers.cpp $(TEST_SRC)/helpers.h
	$(NVCC) $(NVCC_FLAGS) -c $(TEST_SRC)/helpers.cpp -o $(BIN_DIR)/tmp/helpers.o

$(BIN_DIR)/tmp/runTests.o:  $(TEST_SRC)/runTests.cpp
	$(NVCC) $(NVCC_FLAGS) -c $(TEST_SRC)/runTests.cpp -o $(BIN_DIR)/tmp/runTests.o

$(BIN_DIR)/flavors-tests: $(TESTS) $(FLAVORS)
	$(NVCC) $(NVCC_FLAGS) -o $(BIN_DIR)/flavors-tests $(FLAVORS) $(TEST)

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