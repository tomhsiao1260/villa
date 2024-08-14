sudo apt install -y s3fs

# Create a directory where the S3 bucket will be mounted
sudo mkdir /mnt/scrolls
sudo mkdir /mnt/annotated-instances

# Note: Replace 'ACCESS_KEY_ID' and 'SECRET_ACCESS_KEY' with your actual AWS credentials
echo ACCESS_KEY_ID:SECRET_ACCESS_KEY > ~/.passwd-s3fs
chmod 600 ~/.passwd-s3fs

# Mount the S3 buckets to the directory
sudo s3fs herculaneum-scrolls /mnt/scrolls -o use_cache=/tmp -o allow_other -o use_path_request_style -o url=https://s3.amazonaws.com -o passwd_file=~/.passwd-s3fs -o uid=1000 -o gid=1000 -o umask=0022 -o mp_umask=0022
sudo s3fs herculaneum-annotated-instances /mnt/annotated-instances -o use_cache=/tmp -o allow_other -o use_path_request_style -o url=https://s3.amazonaws.com -o passwd_file=~/.passwd-s3fs -o uid=1000 -o gid=1000 -o umask=0022 -o mp_umask=0022

# Add the mount to /etc/fstab for automatic mounting on reboot
# This ensures that the S3 bucket is mounted automatically after a reboot
echo "s3fs#herculaneum-scrolls /mnt/scrolls fuse.s3fs _netdev,allow_other,use_cache=/tmp,umask=0022,uid=1000,gid=1000,url=https://s3.amazonaws.com,passwd_file=/home/ec2-user/.passwd-s3fs 0 0" | sudo tee -a /etc/fstab
echo "s3fs#herculaneum-annotated-instances /mnt/annotated-instances fuse.s3fs _netdev,allow_other,use_cache=/tmp,umask=0022,uid=1000,gid=1000,url=https://s3.amazonaws.com,passwd_file=/home/ec2-user/.passwd-s3fs 0 0" | sudo tee -a /etc/fstab

# Ensure Conda is available
MINICONDA_PREFIX=/usr/local/miniconda
eval "$($MINICONDA_PREFIX/bin/conda shell.bash hook)"
source ~/.bashrc
export PATH=$MINICONDA_PREFIX/bin:$PATH

# Vesuvius
echo "Installing vesuvius scaffold..."
conda create -n vesuvius python=3.12 -y || log_and_exit "Failed to create Vesuvius environment"
conda activate vesuvius
conda install jupyter matplotlib
pip install vesuvius==0.1.4
vesuvius.accept_terms --yes

echo "Installation complete"