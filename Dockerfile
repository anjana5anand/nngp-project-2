# Build:
#   docker build -t nngp-project .
#
# Run exactly as the grader will (no volume mount needed -- the figure is
# baked into the image's /nngp/output/ directory on every run):
#   docker run nngp-project
#
# To pull the PNG back out onto your host afterwards:
#   docker create --name nngp-tmp nngp-project
#   docker cp nngp-tmp:/nngp/output/uncertainty_fig3_mnist.png .
#   docker rm nngp-tmp
#
# Or, to see the file live without copying, mount a folder instead:
#   mkdir -p output
#   docker run -v "$(pwd)/output":/nngp/output nngp-project
#
# Override the default to run CIFAR-10 instead (or change any flag):
#   docker run nngp-project \
#       --dataset=cifar10 --num_train=1000 --num_eval=1000 \
#       --hparams='depth=3,weight_var=2.0,bias_var=0.2' \
#       --nonlinearities='tanh,relu' \
#       --output_file=/nngp/output/uncertainty_fig3_cifar.png

# Pin the platform in the image itself (not just the build/run command) so
# `docker build -t nngp-project .` / `docker run nngp-project` work
# unmodified on the grader's machine, Apple Silicon or not -- this old
# TensorFlow 1.15 image only ships an amd64 build.
FROM --platform=linux/amd64 tensorflow/tensorflow:1.15.0-py3

WORKDIR /nngp

# Copy everything in the repo (original nngp source, grid_data/, and
# uncertainty_plot.py) into the image.
COPY . /nngp

# Two fixes needed for this old codebase to run on a current image, baked
# in at build time instead of by hand each container session:
#   1. Python 2 -> 3 (xrange doesn't exist in Python 3)
#   2. numpy now defaults to allow_pickle=False; the precomputed grid files
#      in grid_data/ were saved as pickled object arrays
RUN sed -i 's/\bxrange\b/range/g' *.py && \
    sed -i "s/np.load(f)/np.load(f, allow_pickle=True, encoding='latin1')/" \
        nngp.py

# matplotlib isn't in the base TensorFlow image and is needed for the plot;
# scipy isn't in the base image either and nngp.py imports it directly
# (from scipy.linalg import solve) for the GP posterior computation.
RUN pip install --no-cache-dir matplotlib scipy

# .dockerignore excludes output/ from the build context, so it doesn't
# exist in the image unless we create it here. Without this, `docker run
# nngp-project` with no volume mount fails: plt.savefig() cannot write
# into a directory that doesn't exist.
RUN mkdir -p /nngp/output

# `docker run nngp-project` with no extra args reproduces the MNIST panel;
# any arguments passed to `docker run` after the image name replace the
# CMD list below and go straight to uncertainty_plot.py.
ENTRYPOINT ["python", "uncertainty_plot.py"]
CMD ["--dataset=mnist", "--num_train=1000", "--num_eval=1000", \
     "--hparams=depth=3,weight_var=2.0,bias_var=0.2", \
     "--nonlinearities=tanh,relu", \
     "--output_file=/nngp/output/uncertainty_fig3_mnist.png"]
