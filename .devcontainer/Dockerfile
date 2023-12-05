FROM mcr.microsoft.com/devcontainers/cpp:0-ubuntu-22.04

# REFRESH!
RUN apt-get update && apt-get upgrade -y && apt-get autoremove -y

# install pip (for cpplint and cmake)
RUN apt-get update && export DEBIAN_FRONTEND=noninteractive \
     && apt-get -y install --no-install-recommends python3-pip

RUN pip install --upgrade cmake


# clang-tidy
RUN apt-get update && export DEBIAN_FRONTEND=noninteractive \
     && apt-get -y install --no-install-recommends clang-tidy clang-format clangd

# install nvm (for npm)
RUN curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.39.5/install.sh | bash
RUN bash /root/.nvm/install.sh

# install npm (using nvm)
RUN bash -i -c "nvm install --lts"

# bazel (using npm)
RUN bash -i -c "npm install -g @bazel/bazelisk"

# install pip and cpplint
RUN apt-get update && export DEBIAN_FRONTEND=noninteractive \
     && apt-get -y install --no-install-recommends python3-pip
# install cpplint into /usr/local/bin/cpplint
RUN pip install cpplint 

# bumpver
RUN python3 -m pip install bumpver

# TEMPORARY FIX TO PYTHON BABEL: https://github.com/python-babel/babel/issues/990
# ValueError: ZoneInfo keys may not be absolute paths, got: /UTC
#RUN rm -f /etc/localtime
#RUN ln -s /usr/share/zoneinfo/UTC /etc/localtime

RUN apt-get update && export DEBIAN_FRONTEND=noninteractive \
     && apt-get -y install --no-install-recommends texlive texlive-latex-extra latexmk texlive-science

# CUDA (too new for clang)
RUN wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-ubuntu2204.pin
RUN sudo mv cuda-ubuntu2204.pin /etc/apt/preferences.d/cuda-repository-pin-600
RUN wget https://developer.download.nvidia.com/compute/cuda/12.3.1/local_installers/cuda-repo-ubuntu2204-12-3-local_12.3.1-545.23.08-1_amd64.deb
RUN sudo dpkg -i cuda-repo-ubuntu2204-12-3-local_12.3.1-545.23.08-1_amd64.deb
RUN sudo cp /var/cuda-repo-ubuntu2204-12-3-local/cuda-*-keyring.gpg /usr/share/keyrings/
RUN sudo apt-get update
RUN sudo apt-get -y install cuda-toolkit-12-3

# CUDA 11 (broken on ubuntu 22)
#RUN wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-ubuntu2004.pin
#RUN sudo mv cuda-ubuntu2004.pin /etc/apt/preferences.d/cuda-repository-pin-600
#RUN wget http://developer.download.nvidia.com/compute/cuda/11.0.2/local_installers/cuda-repo-ubuntu2004-11-0-local_11.0.2-450.51.05-1_amd64.deb
#RUN sudo dpkg -i cuda-repo-ubuntu2004-11-0-local_11.0.2-450.51.05-1_amd64.deb
#RUN sudo apt-key add /var/cuda-repo-ubuntu2004-11-0-local/7fa2af80.pub
#RUN sudo apt-get update
#RUN sudo apt-get -y install cuda -f 

# CUDA 11.5
#RUN wget https://developer.download.nvidia.com/compute/cuda/11.5.0/local_installers/cuda_11.5.0_495.29.05_linux.run
#RUN chmod +x cuda_11.5.0_495.29.05_linux.run
#RUN sudo sh cuda_11.5.0_495.29.05_linux.run --silent --toolkit --toolkitpath=/usr/local/cuda-11.5

# clang-13
RUN apt-get update && export DEBIAN_FRONTEND=noninteractive \
     && apt-get -y install --no-install-recommends clang-13 clangd-13

# clang-15
RUN apt-get update && export DEBIAN_FRONTEND=noninteractive \
     && apt-get -y install --no-install-recommends clang-15 clangd-15

# default clang-13 (compatibility with cuda 11.5)
#RUN sudo update-alternatives --install /usr/bin/clang clang /usr/bin/clang-13 10
#RUN sudo update-alternatives --install /usr/bin/clang++ clang++ /usr/bin/clang++-13 10
#RUN sudo update-alternatives --install /usr/bin/clangd clangd /usr/bin/clangd-13 10

# default clang-15
RUN sudo update-alternatives --install /usr/bin/clang clang /usr/bin/clang-15 10
RUN sudo update-alternatives --install /usr/bin/clang++ clang++ /usr/bin/clang++-15 10
RUN sudo update-alternatives --install /usr/bin/clangd clangd /usr/bin/clangd-15 10

# OpenMP
RUN apt-get update && export DEBIAN_FRONTEND=noninteractive \
     && apt-get -y install --no-install-recommends libomp-dev
