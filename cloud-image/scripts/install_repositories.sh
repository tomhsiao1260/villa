#!/bin/bash

# Function to log and exit on error
log_and_exit() {
    echo "$1"
    exit 1
}

# Ensure Docker is running
echo "Checking Docker service status..."
sudo systemctl status docker || log_and_exit "Docker service is not active. Attempting to start Docker service..."

echo "Attempting to start Docker service..."
sudo systemctl start docker || log_and_exit "Failed to start Docker service. Exiting."

echo "Checking Docker service status again..."
sudo systemctl status docker || log_and_exit "Docker service is not active after start attempt. Exiting."

echo "Docker service is running."

# Ensure Docker socket permissions
echo "Ensuring Docker socket permissions..."
sudo chown root:docker /var/run/docker.sock || log_and_exit "Failed to change Docker socket ownership. Exiting."
sudo chmod 660 /var/run/docker.sock || log_and_exit "Failed to set Docker socket permissions. Exiting."
echo "Docker socket permissions set correctly."

# Check disk space
df -h

# Ensure Conda is available
MINICONDA_PREFIX=/usr/local/miniconda
eval "$($MINICONDA_PREFIX/bin/conda shell.bash hook)"
source ~/.bashrc
export PATH=$MINICONDA_PREFIX/bin:$PATH

# Install gdown
echo "Installing gdown..."
pip install gdown || log_and_exit "Failed to install gdown. Exiting."

# Check disk space
df -h

# Install volume-cartographer using Docker
echo "Installing volume-cartographer using Docker..."
git clone https://github.com/educelab/volume-cartographer.git || log_and_exit "Failed to clone volume-cartographer repository. Exiting."
cd volume-cartographer || log_and_exit "Failed to change directory to volume-cartographer. Exiting."
sudo docker pull ghcr.io/educelab/volume-cartographer:latest || log_and_exit "Failed to pull Docker image for volume-cartographer. Exiting."
cd ..

# Check disk space
df -h

# Install khartes using Conda
echo "Installing khartes using Conda..."
git clone https://github.com/KhartesViewer/khartes.git || log_and_exit "Failed to clone khartes repository. Exiting."
cd khartes || log_and_exit "Failed to change directory to khartes. Exiting."
conda create -n khartes_env python=3.11 -y || log_and_exit "Failed to create Conda environment for khartes. Exiting."
conda activate khartes_env
conda install glib=2.69.1 -y
conda install opencv pyqt tifffile zarr scipy pyopengl -y
conda install pynrrd rectpack -c conda-forge -y || log_and_exit "Failed to install Conda packages for khartes. Exiting."
python setup.py install || log_and_exit "Failed to install khartes. Exiting."
conda deactivate
cd ..

# Clean up Conda caches
conda clean --all -f -y

# Check disk space
df -h

# Install ThaumatoAnakalyptor using Docker and download checkpoints
echo "Installing ThaumatoAnakalyptor using Docker..."
git clone --recurse-submodules https://github.com/schillij95/ThaumatoAnakalyptor.git || log_and_exit "Failed to clone ThaumatoAnakalyptor repository. Exiting."
cd ThaumatoAnakalyptor || log_and_exit "Failed to change directory to ThaumatoAnakalyptor. Exiting."

# Download checkpoints
gdown --id 1gO8Nf4sCaA7r4dO6ePtt0SE0E5ePXSid -O mask3d/saved/train/last-epoch.ckpt || log_and_exit "Failed to download checkpoint 1 for ThaumatoAnakalyptor. Exiting."
gdown --id 1rn3GMOvtJRMBHOxVhWFVSY6IVI6xUnYp -O Vesuvius-Grandprize-Winner/timesformer_wild15_20230702185753_0_fr_i3depoch=12.ckpt || log_and_exit "Failed to download checkpoint 2 for ThaumatoAnakalyptor. Exiting."

# Clean up unnecessary files
rm -rf mask3d/saved/train/last-epoch.ckpt
rm -rf Vesuvius-Grandprize-Winner/timesformer_wild15_20230702185753_0_fr_i3depoch=12.ckpt

sudo apt-get install -y nvidia-docker2 nvidia-container-runtime || log_and_exit "Failed to install NVIDIA Docker dependencies. Exiting."
sudo systemctl restart docker || log_and_exit "Failed to restart Docker service after installing NVIDIA dependencies. Exiting."

sudo docker build -t thaumato_image -f DockerfileThaumato . || log_and_exit "Failed to build Docker image for ThaumatoAnakalyptor. Exiting."
cd ..

# Check disk space
df -h

# Install Crackle-Viewer
echo "Installing Crackle-Viewer..."
git clone https://github.com/schillij95/Crackle-Viewer.git || log_and_exit "Failed to clone Crackle-Viewer repository. Exiting."
cd Crackle-Viewer || log_and_exit "Failed to change directory to Crackle-Viewer. Exiting."
sudo apt-get install -y python3-tk || log_and_exit "Failed to install python3-tk. Exiting."
conda create -n crackle_env python=3.8 -y || log_and_exit "Failed to create Conda environment for Crackle-Viewer. Exiting."
conda activate crackle_env || log_and_exit "Failed to activate Conda environment for Crackle-Viewer. Exiting."
pip install -r requirements.txt || log_and_exit "Failed to install requirements for Crackle-Viewer. Exiting."
conda deactivate
cd ..

# Clean up Conda caches
conda clean --all -f -y

# Check disk space
df -h

# Install Meshlab
echo "Installing Meshlab..."
sudo apt-get update || log_and_exit "Failed to update package list for Meshlab. Exiting."
sudo apt-get install -y meshlab || log_and_exit "Failed to install Meshlab. Exiting."

# Check disk space
df -h

# Install CloudCompare
echo "Installing CloudCompare..."
sudo add-apt-repository ppa:cloudcompare/trunk -y || log_and_exit "Failed to add CloudCompare repository. Exiting."
sudo apt-get update || log_and_exit "Failed to update package list for CloudCompare. Exiting."
sudo apt-get install -y cloudcompare || log_and_exit "Failed to install CloudCompare. Exiting."

# Check disk space
df -h

echo "Installation completed!"
