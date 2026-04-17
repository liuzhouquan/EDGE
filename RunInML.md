Our servers are equipped with the Slurm job scheduler, which efficiently manages and allocates resources across multiple users and jobs. Here’s a short tutorial to submit your GPU jobs. You are required to submit jobs via slurm. Do not run training, computation, or jupyter notebook directly in the SSH shell. They may fail silently.

a. Create a .sh file (e.g., run_training.sh) with the following structure:
#!/bin/bash

# ======== SLURM Job Configuration ========
#SBATCH --job-name=<job_name>             # [REQUIRED] Name of the job as it appears in the job queue (e.g., "AAAI-training")
#SBATCH --time=00:30:00                   # [REQUIRED] Wall time limit in HH:MM:SS (adjust to your expected runtime)
#SBATCH --open-mode=append                # Append to output and error logs instead of overwriting
#SBATCH --output=<output_file>.log        # [RECOMMENDED] File to write standard output (e.g., "DnCNN-output.log")
#SBATCH --error=<error_file>.log          # [RECOMMENDED] File to write standard error (e.g., "DnCNN-error.log")
#SBATCH --gres=gpu:1                      # [REQUIRED] Request 1 GPU, If you requests more than 1, you need to get approval from the Slack channel bot)

# ======== Job Execution Steps ========

# Navigate to the working directory where your code and virtual environment are located
cd /data/<your_username>/<project_path>

# Activate the Python virtual environment (adjust if it's named differently)
source venv/bin/activate

# Run your Python script or other commands
python3 train.py 

b. Use the sbatch command to send your job to the SLURM scheduler:

sbatch run_training.sh

This lets SLURM find the right time and resources to run your job, efficient and fair for all users.

c. Monitor Your Jobs via the squeue command.

squeue -u $USER