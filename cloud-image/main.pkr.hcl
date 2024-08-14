source "amazon-ebs" "small_instance" {
  ami_name      = var.ami_name
  instance_type = var.aws_instance_type
  region        = var.aws_region

  # This will find and use the latest Ubuntu 20.04 image
  #  TODO: We will want to change that to a GPU-instance base image later

  source_ami_filter {
    filters = {
      name                = "ubuntu/images/hvm-ssd/ubuntu-focal-20.04-amd64-server-*"
      root-device-type    = "ebs"
      virtualization-type = "hvm"
    }
    most_recent = true
    owners      = ["099720109477"] # Canonical
  }
  
  #Uncomment this and comment before for GPU Instance
  #source_ami_filter {
  #  filters = {
  #    name                = "Deep Learning AMI GPU PyTorch*"
  #    root-device-type    = "ebs"
  #    virtualization-type = "hvm"
  #  }
  #  most_recent = true
  #  owners      = ["898082745236"] # AWS Deep Learning AMIs
  #}

  ssh_username = "ubuntu"
  # TODO: Uncomment the below when we want to make the AMI publicly available
  # ami_groups   = ["all"]  # Make the AMI public

  launch_block_device_mappings {
    device_name = "/dev/sda1"
    # TODO: change this accordingly
    volume_size           = 30 # in GB, ~0.08 USD/GB/month... this is never loaded?
    delete_on_termination = true
    volume_type           = "gp3"
  }
}

build {
  sources = ["source.amazon-ebs.small_instance"]

  # Run the set up script

  provisioner "shell" {
    inline = [
      "sudo apt-get update",
      "sudo apt-get install -y cloud-guest-utils",
      "echo 'df -h'",
      "df -h",
      "echo 'lsblk'",
      "lsblk"
    ]
  }

  provisioner "shell" {
    script = "scripts/install_dependencies.sh"
  }

  #provisioner "shell" {
  #  script = "scripts/install_repositories.sh"
  #}

  provisioner "shell" {
    script = "scripts/load_s3disk.sh"
  }

  # Optionally, add more scripts to run here...
}