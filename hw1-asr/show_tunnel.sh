#!/bin/bash

# ================= Configuration =================
# [Optional] Enter the public address you use to SSH into the cluster.
# If left empty, the script will try to use the current hostname.
EXTERNAL_LOGIN_HOST="" 
# =================================================

# 1. Define Colors
GREEN='\033[0;32m'
CYAN='\033[0;36m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# 2. CHECK PORT (Strict Requirement)
# If argument $1 is empty (-z), print error and exit.
if [ -z "$1" ]; then
    echo -e "${RED}❌ Error: Missing Port Number!${NC}"
    echo -e "Usage:   ${CYAN}$0 <PORT>${NC}"
    echo -e "Example: ${CYAN}$0 8888${NC}"
    exit 1
fi

PORT=$1

# 3. Automatically detect the running compute node
# squeue arguments: --me (my jobs), --state=RUNNING, -h (no header), -o %N (node name)
NODE=$(squeue --me --state=RUNNING -h -o %N | head -n 1)

# 4. Check if a node was found
if [ -z "$NODE" ]; then
    echo -e "${RED}❌ Error: No RUNNING job found.${NC}"
    echo "Please ensure you have started a job via 'sbatch' or 'srun'."
    exit 1
fi

# 5. Determine the Login Node Hostname
if [ -z "$EXTERNAL_LOGIN_HOST" ]; then
    LOGIN_HOST=$(hostname -f)
else
    LOGIN_HOST=$EXTERNAL_LOGIN_HOST
fi

# 6. Generate the SSH Command
SSH_CMD="ssh -N -f -L ${PORT}:${NODE}:${PORT} ${USER}@${LOGIN_HOST}"

# 7. Print the Output
echo "------------------------------------------------------------------"
echo -e "✅ Job Status:   ${GREEN}RUNNING${NC}"
echo -e "🖥️  Compute Node: ${CYAN}${NODE}${NC}"
echo -e "🔌 Target Port:  ${YELLOW}${PORT}${NC}"
echo "------------------------------------------------------------------"
echo -e "👇 Copy the command below and run it on your LOCAL PC terminal:"
echo ""
echo -e "${CYAN}${SSH_CMD}${NC}"
echo ""
echo "------------------------------------------------------------------"
echo -e "🔗 After connecting, visit: http://localhost:${PORT}"