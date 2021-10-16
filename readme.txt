if this error appears,
  ERROR: Could not find a version that satisfies the requirement tensorflow
  ERROR: No matching distribution found for tensorflow
  Downgrade python by (new environment)
    py -3.7 -m venv venv
    pip install -r requirements.txt

if this error appears
  tensorflow/stream_executor/platform/default/dso_loader.cc:60]
  Could not load dynamic library 'cudart64_110.dll'; dlerror:
  cudart64_110.dll not found 2021-03-16 21:33:05.397514: I
  tensorflow/stream_executor/cuda/cudart_stub.cc:29]
  Ignore above cudart dlerror if you do not have a GPU set up on your machine.
    see here
      https://stackoverflow.com/questions/59823283/could-not-load-dynamic-library-cudart64-101-dll-on-tensorflow-cpu-only-install
      https://ourcodeworld.com/articles/read/1433/how-to-fix-tensorflow-warning-could-not-load-dynamic-library-cudart64-110dll-dlerror-cudart64-110dll-not-found
    Install version 11.2
    SHould be OK.

to get the code to load cifar-10 from directory (instead of cifar-10.loaddata())
  get the files from here.. https://github.com/snatch59/load-cifar-10
  use load_cifar_10_alt

for tensorflow GPU support, read here
  https://towardsdatascience.com/installing-tensorflow-with-cuda-cudnn-and-gpu-support-on-windows-10-60693e46e781

if installing tensorflow-gpu receives --user ERROR
  https://stackoverflow.com/questions/32167418/python-pip-install-trouble-shooting-permissionerror-winerror-5-access-is
  "Since I had already tried a first time without running the cmd prompt as admin, in my c:\Users\"USER"\AppData\Local\Temp folder
  I found it was trying to run files from the same pip-u2e7e0ad-uninstall folder.
  Deleting this folder from the Temp folder and retrying the installation fixed the issue for me."

if get this ERROR
  Cannot dlopen some GPU libraries. Please make sure the missing libraries
  mentioned above are installed properly if you would like to use GPU.
  Follow the guide at https://www.tensorflow.org/install/gpu for how to download and setup
  the required libraries for your platform.
  https://www.codingforentrepreneurs.com/blog/install-tensorflow-gpu-windows-cuda-cudnn/

to check if gpu is available
  # Check that GPU is available: cf. https://colab.research.google.com/notebooks/gpu.ipynb
  assert(tf.test.is_gpu_available())

how to toggle between GPU and CPU
  https://stackoverflow.com/questions/62279112/for-tensorflow-2-x-how-to-switch-between-the-cpu-and-gpu-version
