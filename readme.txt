Readme:
===================================================================
I use python3.6 and latest Tensorflow source version (master branch at August 3th, 2017).

python3 at: /usr/bin/python3

pathon3 lib: /usr/local/lib/python3.6/site-packages

====================================================================
tensorflow configure process:
[putamen:~/Projects/tensorflow] hxie1% ./configure
WARNING: Running Bazel server needs to be killed, because the startup options are different.
Please specify the location of python. [Default is /usr/bin/python]:  /usr/bin/python3
Found possible Python library paths:
/Library/Python/2.7/site-packages
Please input the desired Python library path to use.  Default is /Library/Python/2.7/site-packages:   /usr/local/lib/python3.6/site-packages
Do you wish to build TensorFlow with Google Cloud Platform support? [y/N]: n
No Google Cloud Platform support will be enabled for TensorFlow.

Do you wish to build TensorFlow with Hadoop File System support? [y/N]: n
No Hadoop File System support will be enabled for TensorFlow.

Do you wish to build TensorFlow with XLA JIT support? [y/N]: n
No XLA JIT support will be enabled for TensorFlow.

Do you wish to build TensorFlow with VERBS support? [y/N]: n
No VERBS support will be enabled for TensorFlow.

Do you wish to build TensorFlow with OpenCL support? [y/N]: n
No OpenCL support will be enabled for TensorFlow.

Do you wish to build TensorFlow with CUDA support? [y/N]: n
No CUDA support will be enabled for TensorFlow.

Do you wish to build TensorFlow with MPI support? [y/N]: n
No MPI support will be enabled for TensorFlow.

Please specify optimization flags to use during compilation when bazel option "--config=opt" is specified [Default is -march=native]:
Add "--config=mkl" to your bazel command to build with MKL support.
Please note that MKL on MacOS or windows is still not supported.
If you would like to use a local MKL instead of downloading, please set the environment variable "TF_MKL_ROOT" every time before build.
Configuration finished
[putamen:~/Projects/tensorflow] hxie1%

============================================================


Bazel compile in putamen:
bazel build --config=opt //tensorflow/tools/pip_package:build_pip_package
or
bazel build --config=opt --config=mkl //tensorflow/tools/pip_package:build_pip_package

# build tensor flow for Argon HPC successfully
bazel build --config=mkl --copt="-DEIGEN_USE_VML" -c opt //tensorflow/tools/pip_package:build_pip_package
bazel build --config=mkl -c opt //tensorflow/tools/pip_package:build_pip_package


==============================================================
Generate pip3 wheel:
[putamen:~/Projects/tensorflow] hxie1% ./bazel-bin/tensorflow/tools/pip_package/build_pip_package ~/temp/tensorflow_pkg

==============================================================

Install wheel:
sudo -H pip3 install ~/temp/tensorflow_pkg/tensorflow-1.3.0rc1-cp36-cp36m-macosx_10_12_x86_64.whl

=========================================

************************************************************
************************************************************


for IntelPython3.5
======================================
python3 at: ~/miniconda3/bin/python3.6

pathon3 lib: /opt/intel/intelpython3/lib

bazel compile:(has compile error in fopenmp)
bazel build --config=mkl --copt=”-DEIGEN_USE_VML” -c opt //tensorflow/tools/pip_package:build_pip_package
Generate pip3 wheel:
bazel-bin/tensorflow/tools/pip_package/build_pip_package ~/path_to_save_wheel
pip install --upgrade --user ~/path_to_save_wheel/wheel_name.whl

Add "--config=mkl" to your bazel command to build with MKL support.
Please note that MKL on MacOS or windows is still not supported.
If you would like to use a local MKL instead of downloading, please set the environment variable "TF_MKL_ROOT" every time before build.
in tcshell fille ~/.cshrc, add below line:
set TF_MKL_ROOT=/opt/intel/intelpython3/bin/tensorflow/third_party/mkl

install intel tensorflow from conda.

conda install -c intel tensorflow

for intel CPU performace  test:
 ~/miniconda3/bin/python3.6 ~/Projects/3DSegTensorFlow/Segment3D.py T1T2LabelCubicNormalize.csv 10 280,240,200,160,120,80,40,26 0.002

************************************************************
************************************************************
Use conda to install intel python

python3 at: /Users/hxie1/miniconda3/bin/python3.5
pathon3 lib: /Users/hxie1/miniconda3/lib

conda search --full-name python
conda install python=3.5.3  # for intel Python3
conda install tensorflow=1.1.0


==============================
Test T1T2Label*.csv files are generated by C++ program in https://github.com/Hui-Xie/NeuralNetworkInDAALandITK.git
===============================


==========================
Argon Server Config:
(root) [hxie1@argon-login-1 tensorflow]$ ./configure
WARNING: Output base '/Users/hxie1/.cache/bazel/_bazel_hxie1/92a1be4b0e8659fe8a492b1123133227' is on NFS. This may lead to surprising failures and undetermined behavior.
You have bazel 0.5.3- (@non-git) installed.
Please specify the location of python. [Default is /Users/hxie1/intel/intelpython3/bin/python]:


Found possible Python library paths:
  /Users/hxie1/intel/intelpython3/lib/python3.5/site-packages
Please input the desired Python library path to use.  Default is [/Users/hxie1/intel/intelpython3/lib/python3.5/site-packages]

Do you wish to build TensorFlow with jemalloc as malloc support? [Y/n]: Y
jemalloc as malloc support will be enabled for TensorFlow.

Do you wish to build TensorFlow with Google Cloud Platform support? [y/N]:
No Google Cloud Platform support will be enabled for TensorFlow.

Do you wish to build TensorFlow with Hadoop File System support? [y/N]:
No Hadoop File System support will be enabled for TensorFlow.

Do you wish to build TensorFlow with XLA JIT support? [y/N]:
No XLA JIT support will be enabled for TensorFlow.

Do you wish to build TensorFlow with GDR support? [y/N]:
No GDR support will be enabled for TensorFlow.

Do you wish to build TensorFlow with VERBS support? [y/N]:
No VERBS support will be enabled for TensorFlow.

Do you wish to build TensorFlow with OpenCL support? [y/N]:
No OpenCL support will be enabled for TensorFlow.

Do you wish to build TensorFlow with CUDA support? [y/N]:
No CUDA support will be enabled for TensorFlow.

Do you wish to build TensorFlow with MPI support? [y/N]:
No MPI support will be enabled for TensorFlow.

Please specify optimization flags to use during compilation when bazel option "--config=opt" is specified [Default is -march=native]:


Add "--config=mkl" to your bazel command to build with MKL support.
Please note that MKL on MacOS or windows is still not supported.
If you would like to use a local MKL instead of downloading, please set the environment variable "TF_MKL_ROOT" every time before build.
Configuration finished


Bazel compile:
(root) [hxie1@argon-login-1 tensorflow]$ bazel --output_user_root="~" build --config=opt //tensorflow/tools/pip_package:build_pip_package

==============================================================
Generate pip3 wheel:
(root) [hxie1@argon-login-1 tensorflow]$ ./bazel-bin/tensorflow/tools/pip_package/build_pip_package ~/temp/tensorflow_pkg

==============================================================

Install wheel:
(root) [hxie1@argon-login-1 tensorflow]$ pip install ~/temp/tensorflow_pkg/tensorflow-1.3.0-cp35-cp35m-linux_x86_64.whl