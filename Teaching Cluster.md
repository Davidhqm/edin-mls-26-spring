Here is the content of the Teaching Cluster document organized into Markdown format. **Recommended: read `Teaching Cluster.pdf` directly** for the authoritative version, since the tutorial includes diagrams and figures.

# Teaching GPU Cluster Overview

## System Architecture

The cluster architecture consists of a Client connecting via SSH to a **Head Node**. The Head Node interfaces with a Shared Distributed File System and AFS (data on DICE machines). From the Head Node, users issue Slurm commands (e.g., `srun`, `sbatch`) to distribute work across **Compute Nodes** (Node 0 to Node N).

---

## Basic Information

**Hardware Specifications**
The teaching cluster has over 120 GPUs distributed across 20 servers (landonia [01-25]) containing:

* 100 NVIDIA GTX 1060 6GB GPUs.


* 8 NVIDIA RTX A6000 48GB GPUs.


* A small number of NVIDIA TITAN-X 12GB GPUs.


* *Note:* As of the latest update, there are also 8 NVIDIA H200 GPUs available.



---

## Access and Usage

### 1. Connecting to the Head Node

Access is managed through SSH and Slurm commands.

* **Prerequisite:** You must be connected to the Informatics VPN or using a DICE machine.


* **Command:**
```bash
ssh <YOUR_UUN>@mlp.inf.ed.ac.uk
```





* **From remote DICE connection:**
If you are on VPN or eduroam, you can SSH into a DICE machine first: `ssh <YOUR_UUN>@ssh.inf.ed.ac.uk`.



### Head Node Warnings

After successfully connecting, you will see a welcome message.

> **Important:** This is a cluster head node. Please do not run compute-intensive processes here. This node is intended to provide an interface to the cluster only. Please nice any long-running processes.
> 
> 

### 2. Running Code with Slurm

There are two ways to run code:

1. **Interactive Job:** Allows you to interact directly with the allocated compute resources. This is good for debugging and testing code.


2. **Batch Job:** Runs the task automatically based on a pre-written script (`.sh` file) without interaction or constant user attention. This is good for long-running tasks.



#### Example: Interactive Job

To run code utilizing 1 GPU interactively:

```bash
srun -p Teaching -w saxa --gres gpu:1 --pty bash
```



After executing this, running `nvidia-smi` will show the allocated GPU details (e.g., NVIDIA H200).

#### Example: Batch Job

To submit a batch script utilizing 1 GPU:

```bash
sbatch --gres=gpu:1 test.sh
```



* You will see a confirmation: "Submitted batch job [ID]".


* The output is saved as `slurm-[ID].out` in the current directory by default.


* You can verify the output using `cat slurm-[ID].out`.

#### Memory Allocation Note (Why is my program killed during model weight loading?)

By default, the Teaching cluster may allocate ~8GB of RAM if ```--mem``` is not explicitly specified.

Therefore, when running benchmark.sh without specifying the memory requirement, you may find your job be killed due to insufficient RAM, especially during model weight loading.

To avoid this out-of-memory issue, please explicitly set the memory requirement when submitting the job. For example:

```bash
srun -p Teaching -w saxa --gres gpu:1 --mem=16G --pty bash
```

or in the sbatch script:

```bash
#SBATCH --mem=16G
```

---

## Resource Allocation & Management

### Requesting Specific GPU Types

You can specify the type of GPU you wish to use.

**Example: Requesting 1 NVIDIA Titan X**

* **Interactive:** `srun --gres=gpu:titan_x:1 --pty bash`.


* **Batch:** `sbatch --gres=gpu:titan_x:1 test.sh`.



**Quotas & Constraints**

* You are allowed to request a maximum of: 8 GTX 1060, 4 Titan X, 1 Titan X Pascal, or 2 A6000 GPUs at a time.


* Use `-p Teaching -w saxa` to access the H200 GPUs.


* **Note:** Please allocate resources according to your specific requirements. Avoid over-allocating to ensure resources are available for others.



### Useful Slurm Commands

* **Check available GPU types:**
```bash
scontrol show node | grep gpu
```


(Displays GRES configurations like `gpu:titan_x_pascal:1`, `gpu:gtx_1060:2`, etc.) .


* **Check current job status:**
```bash
squeue
```


(Displays JOBID, PARTITION, USER, ST (status), NODES, etc.).


* **Cancel a job:**
```bash
scancel <job_id>
```




---

## How to write code in the teaching cluster (VS Code Guide)

The recommended method for writing code is using **VS Code** with the **Remote - SSH** extension.

### Installation & Configuration

1. Open VS Code.


2. Go to **Extensions**.


3. Search for "Remote" and install the **"Remote - SSH"** extension.


4. Once installed, the **Remotes icon** (computer monitor symbol) will appear on the left sidebar.


5. Click the Remotes icon, then select the **Config icon** (gear) located to the right of "SSH".


6. Choose the configuration file ending in `.ssh/config`.


7. In the config file, enter the following text and save it:


```text
Host teaching-cluster
    HostName mlp.inf.ed.ac.uk
    User <YOUR_UUN>
```



### Connecting

8. Refresh the **REMOTES** page. You will see "teaching-cluster" listed.


9. Connect to the Informatics VPN or use a DICE machine.


10. Click the right arrow icon next to `teaching-cluster` to connect.


11. Enter your DICE account password when prompted.


12. Once connected (green indicator), click **"Open Folder"** and select your home directory (`/home/<YOUR_UUN>`).



### Important Workflow Notes

* **Synchronization** 
  * Any changes made in VS Code (creating files, writing code) automatically synchronize with the cluster. This is why VS Code is recommended.


* **Head Node Restrictions**
  * Only install your environment and write code on the head node.


  * Perform Git operations on the head node.


  * **Do not run code on the head node.** It has limited resources and no GPUs. Running `torch.cuda.is_available()` on the head node will return `False`.




  * **Compute Nodes:** Only run your code on compute nodes using `srun` or `sbatch`.



---

### Support

For more details about the school cluster, file storage, and preventing data loss, refer to:
[https://computing.help.inf.ed.ac.uk/teaching-cluster](https://computing.help.inf.ed.ac.uk/teaching-cluster).
