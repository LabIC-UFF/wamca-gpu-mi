################################################################################
# Automatically-generated file. Do not edit!
################################################################################

# Add inputs and outputs from these tool invocations to the build variables 
CPP_SRCS += \
../source/except.cpp \
../source/log.cpp \
../source/main.cpp \
../source/mlads.cpp \
../source/mlgputask.cpp \
../source/mlkernel.cpp \
../source/mlparser.cpp \
../source/mlproblem.cpp \
../source/mlsearch.cpp \
../source/mlsolution.cpp \
../source/mlsolver.cpp \
../source/mltask.cpp \
../source/mtrand.cpp \
../source/tsplib.cpp \
../source/utils.cpp 

CU_SRCS += \
../source/mlk2opt.cu \
../source/mlkoropt.cu \
../source/mlkswap.cu \
../source/mlkutil.cu 

CU_DEPS += \
./source/mlk2opt.d \
./source/mlkoropt.d \
./source/mlkswap.d \
./source/mlkutil.d 

OBJS += \
./source/except.o \
./source/log.o \
./source/main.o \
./source/mlads.o \
./source/mlgputask.o \
./source/mlk2opt.o \
./source/mlkernel.o \
./source/mlkoropt.o \
./source/mlkswap.o \
./source/mlkutil.o \
./source/mlparser.o \
./source/mlproblem.o \
./source/mlsearch.o \
./source/mlsolution.o \
./source/mlsolver.o \
./source/mltask.o \
./source/mtrand.o \
./source/tsplib.o \
./source/utils.o 

CPP_DEPS += \
./source/except.d \
./source/log.d \
./source/main.d \
./source/mlads.d \
./source/mlgputask.d \
./source/mlkernel.d \
./source/mlparser.d \
./source/mlproblem.d \
./source/mlsearch.d \
./source/mlsolution.d \
./source/mlsolver.d \
./source/mltask.d \
./source/mtrand.d \
./source/tsplib.d \
./source/utils.d 


# Each subdirectory must supply rules for building sources it contributes
source/%.o: ../source/%.cpp
	@echo 'Building file: $<'
	@echo 'Invoking: NVCC Compiler'
	/usr/local/cuda-7.5/bin/nvcc -D_FORCE_INLINES -DLOG_LEVEL=1 -DGPU_PROFILE -DMLP_CPU_ADS -DMLP_GPU_ADS -I/home/eyder/workspace/cuda/include -include /home/eyder/workspace/cuda/wamca2016/source/directives.h -lineinfo -O3 -Xcudafe --diag_suppress=expr_has_no_effect -Xptxas -v -gencode arch=compute_35,code=sm_35  -odir "source" -M -o "$(@:%.o=%.d)" "$<"
	/usr/local/cuda-7.5/bin/nvcc -D_FORCE_INLINES -DLOG_LEVEL=1 -DGPU_PROFILE -DMLP_CPU_ADS -DMLP_GPU_ADS -I/home/eyder/workspace/cuda/include -include /home/eyder/workspace/cuda/wamca2016/source/directives.h -lineinfo -O3 -Xcudafe --diag_suppress=expr_has_no_effect -Xptxas -v --compile  -x c++ -o  "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '

source/%.o: ../source/%.cu
	@echo 'Building file: $<'
	@echo 'Invoking: NVCC Compiler'
	/usr/local/cuda-7.5/bin/nvcc -D_FORCE_INLINES -DLOG_LEVEL=1 -DGPU_PROFILE -DMLP_CPU_ADS -DMLP_GPU_ADS -I/home/eyder/workspace/cuda/include -include /home/eyder/workspace/cuda/wamca2016/source/directives.h -lineinfo -O3 -Xcudafe --diag_suppress=expr_has_no_effect -Xptxas -v -gencode arch=compute_35,code=sm_35  -odir "source" -M -o "$(@:%.o=%.d)" "$<"
	/usr/local/cuda-7.5/bin/nvcc -D_FORCE_INLINES -DLOG_LEVEL=1 -DGPU_PROFILE -DMLP_CPU_ADS -DMLP_GPU_ADS -I/home/eyder/workspace/cuda/include -include /home/eyder/workspace/cuda/wamca2016/source/directives.h -lineinfo -O3 -Xcudafe --diag_suppress=expr_has_no_effect -Xptxas -v --compile --relocatable-device-code=false -gencode arch=compute_35,code=compute_35 -gencode arch=compute_35,code=sm_35  -x cu -o  "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '


