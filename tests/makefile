CCX=/usr/local/cuda-8.0/bin/nvcc
CCX_STD=--coverage -g -O0 --std=c++11
CCX_ARCH=-D_FORCE_INLINES -DLOG_LEVEL=1 -DGPU_PROFILE -O3 -gencode arch=compute_35,code=sm_35 -gencode arch=compute_50,code=sm_50
NVCC_INFO=-Xcudafe --diag_suppress=expr_has_no_effect -Xptxas -O3,-v  # passing O3 to nvcc too
NVCC_TEST=-Xcompiler --coverage  -g  #--compiler-option --coverage
SRC_PATH=../source
LIBS_PATH=../libs
# gtest
GTEST_PATH=$(LIBS_PATH)/gtestlib/build/googletest
GMOCK_PATH=$(LIBS_PATH)/gtestlib/build/googlemock
GTEST_INCLUDE=$(LIBS_PATH)/gtestlib/googletest/include
GMOCK_INCLUDE=$(LIBS_PATH)/gtestlib/googlemock/include

all:
	@echo "please type 'make test'"

test: clean wamca2016.test
	@echo "Executing tests"
	./wamca2016.test

#/usr/local/cuda-8.0/bin/nvcc -D_FORCE_INLINES -DLOG_LEVEL=1 -DGPU_PROFILE -O3 -gencode arch=compute_35,code=sm_35 -gencode arch=compute_50,code=sm_50  -odir "source" -M -o "source/log.d" "../source/log.cpp"
# (CCX_ARCH)  -odir "source" -M -o "source/log.d" "../source/log.cpp"

#/usr/local/cuda-8.0/bin/nvcc -g -Xcudafe --diag_suppress=expr_has_no_effect -Xptxas -v  -D_FORCE_INLINES -DLOG_LEVEL=1 -DGPU_PROFILE  -O3 --compile  -x c++ -o  "source/log.o" "../source/log.cpp"


wamca2016.test : wamca2016.Test.cpp
	#g++ $(CCX_STD)
	#$(CCX) $(CCX_ARCH) $(NVCC_INFO) --coverage -g -O0 --std=c++11
	$(CCX) $(CCX_ARCH) $(NVCC_INFO) $(NVCC_TEST) --std=c++11 \
	-I$(SRC_PATH) -I$(GTEST_INCLUDE) -I$(GMOCK_INCLUDE) \
	 $< -o $@ \
	$(GMOCK_PATH)/gtest/libgtest_main.a $(GMOCK_PATH)/gtest/libgtest.a $(GMOCK_PATH)/libgmock.a \
	$(SRC_PATH)/mtrand.cpp \
	$(SRC_PATH)/mlproblem.cpp $(SRC_PATH)/tsplib.cpp $(SRC_PATH)/except.cpp $(SRC_PATH)/utils.cpp \
	$(SRC_PATH)/mlsolution.cpp $(SRC_PATH)/mlads.cpp \
	$(SRC_PATH)/mlkswap.cu $(SRC_PATH)/mlkernel.cu \
	 -lpthread -ldl

test-coverage:
	mkdir -p reports
	lcov --directory . --capture --output-file reports/app.info
	(cd reports && genhtml app.info)

# cleaning tests
clean:
	rm -f *.test
	rm -f *.gcda
	rm -f *.gcno
	rm -f *.info
