
PROJECT = wamca2016

all: bazel # release-2018

bazel:
	bazel clean
	# -@rules_cuda//cuda:archs=compute_61:compute_61,sm_61
	# bazel build ... 2> out.txt
	# bazel build --@rules_cuda//cuda:archs=compute_86:compute_86,sm_86 ... 2> out.txt
	bazel build --@rules_cuda//cuda:archs=compute_75:compute_75,sm_75 ... 2> out.txt
	# bazel build --@rules_cuda//cuda:archs=compute_35:compute_35,sm_35 ... 2> out.txt
	# bazel build --@rules_cuda//cuda:archs=compute_50:compute_50,sm_50 ... 2> out.txt
	# cat out.txt | grep registers -B 4 | sed 's/^/-\n/'
	cat out.txt | grep registers -B 4 > out-registers.txt

release:
	make -C Release $(PROJECT)

clean:
	make -C Release $(PROJECT) clean

release-2018:
	make -C Release-cuda-8.0 
