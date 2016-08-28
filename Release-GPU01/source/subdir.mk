################################################################################
# Automatically-generated file. Do not edit!
################################################################################

# Add inputs and outputs from these tool invocations to the build variables 
CPP_SRCS += \
../source/except.cpp \
../source/log.cpp \
../source/mlads.cpp \
../source/mlproblem.cpp \
../source/mlsolution.cpp \
../source/mtrand.cpp \
../source/tsplib.cpp \
../source/utils.cpp 

CU_SRCS += \
../source/main.cu \
../source/mlk2opt.cu \
../source/mlkernel.cu \
../source/mlkoropt.cu \
../source/mlkswap.cu \
../source/mlkutil.cu 

CU_DEPS += \
./source/main.d \
./source/mlk2opt.d \
./source/mlkernel.d \
./source/mlkoropt.d \
./source/mlkswap.d \
./source/mlkutil.d 

OBJS += \
./source/except.o \
./source/log.o \
./source/main.o \
./source/mlads.o \
./source/mlk2opt.o \
./source/mlkernel.o \
./source/mlkoropt.o \
./source/mlkswap.o \
./source/mlkutil.o \
./source/mlproblem.o \
./source/mlsolution.o \
./source/mtrand.o \
./source/tsplib.o \
./source/utils.o 

CPP_DEPS += \
./source/except.d \
./source/log.d \
./source/mlads.d \
./source/mlproblem.d \
./source/mlsolution.d \
./source/mtrand.d \
./source/tsplib.d \
./source/utils.d 


# Each subdirectory must supply rules for building sources it contributes
source/%.o: ../source/%.cpp
	@echo 'Building file: $<'
	@echo 'Invoking: NVCC Compiler'
	/usr/local/cuda-7.5/bin/nvcc -D_FORCE_INLINES -DLOG_LEVEL=1 -DGPU_PROFILE -DMLP_CPU_ADS -DMLP_GPU_ADS -lineinfo -O3 -Xcudafe --diag_suppress=expr_has_no_effect -Xptxas -v -gencode arch=compute_35,code=sm_35  -odir "source" -M -o "$(@:%.o=%.d)" "$<"
	/usr/local/cuda-7.5/bin/nvcc -D_FORCE_INLINES -DLOG_LEVEL=1 -DGPU_PROFILE -DMLP_CPU_ADS -DMLP_GPU_ADS -lineinfo -O3 -Xcudafe --diag_suppress=expr_has_no_effect -Xptxas -v --compile  -x c++ -o  "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '

source/%.o: ../source/%.cu
	@echo 'Building file: $<'
	@echo 'Invoking: NVCC Compiler'
	/usr/local/cuda-7.5/bin/nvcc -D_FORCE_INLINES -DLOG_LEVEL=1 -DGPU_PROFILE -DMLP_CPU_ADS -DMLP_GPU_ADS -lineinfo -O3 -Xcudafe --diag_suppress=expr_has_no_effect -Xptxas -v -gencode arch=compute_35,code=sm_35  -odir "source" -M -o "$(@:%.o=%.d)" "$<"
	/usr/local/cuda-7.5/bin/nvcc -D_FORCE_INLINES -DLOG_LEVEL=1 -DGPU_PROFILE -DMLP_CPU_ADS -DMLP_GPU_ADS -lineinfo -O3 -Xcudafe --diag_suppress=expr_has_no_effect -Xptxas -v --compile --relocatable-device-code=false -gencode arch=compute_35,code=compute_35 -gencode arch=compute_35,code=sm_35  -x cu -o  "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '


