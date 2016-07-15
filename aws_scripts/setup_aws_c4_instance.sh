#!/usr/bin
echo "Setting up AWS C4 instance..."

# Install libraries and Python packages
sudo apt-get update
sudo apt-get upgrade
sudo apt-get install git python-pip python-dev gfortran
sudo pip install dropbox pandas scipy scikit-learn
sudo pip install git+https://github.com/fmfn/BayesianOptimization.git

# Prepare TensorFlow installation
export TF_BINARY_URL=https://storage.googleapis.com/tensorflow/linux/cpu/tensorflow-0.9.0-cp27-none-linux_x86_64.whl
echo "export TF_BINARY_URL=https://storage.googleapis.com/tensorflow/linux/cpu/tensorflow-0.9.0-cp27-none-linux_x86_64.whl" >> .profile

# Install TensorFlow
sudo pip install --upgrade $TF_BINARY_URL

echo "AWS C4 instance setup!"