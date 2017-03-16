#!/bin/bash
# Reference: https://www.tensorflow.org/install/install_sources

sudo apt-get update
sudo apt-get upgrade -y

# Install bazel (https://bazel.build/versions/master/docs/install.html#ubuntu)
echo "deb [arch=amd64] http://storage.googleapis.com/bazel-apt stable jdk1.8" | sudo tee /etc/apt/sources.list.d/bazel.list
curl https://bazel.build/bazel-release.pub.gpg | sudo apt-key add -
sudo apt-get update && sudo apt-get install -y bazel
sudo apt-get upgrade bazel

# Install Python dependencies
sudo apt-get install -y python-numpy python-dev python-pip python-wheel

# Get latest TensorFlow source code
git clone https://github.com/tensorflow/tensorflow

cd tensorflow

# Configure installation
./configure
# Please specify the location of python. [Default is /usr/bin/python]: /usr/bin/python2.7
# The rest set to default

# Build and install the pip package
bazel build --config=opt //tensorflow/tools/pip_package:build_pip_package
bazel-bin/tensorflow/tools/pip_package/build_pip_package /tmp/tensorflow_pkg
sudo pip install /tmp/tensorflow_pkg/*.whl

# Set Dropbox API secret key
export DROPBOX_SECRET_KEY="qECz4Lio64gAAAAAAAADKCBiIafW0-teoaxb7jaNJVjcn517S7mH0l7rwjZXbThX"
echo $DROPBOX_SECRET_KEY

# Get latest code
git clone https://github.com/grantathon/barchagranda.git
