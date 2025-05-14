# run sh command
import subprocess
import sys

def run_sh_command(command):
    """
    Run a shell command and return the output.
    """
    process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    stdout, stderr = process.communicate()
    if process.returncode != 0:
        print(f"Error: {stderr.decode('utf-8')}")
        sys.exit(1)
    return stdout.decode('utf-8')

# pqn_jaxtari_k4_hier_comb.yaml
# pqn_jaxtari_k5_hier_comb_pre.yaml
run_sh_command("CUDA_VISIBLE_DEVICES=12 uv run main.py +alg=pqn_jaxtari_k4_hier_comb")
print("pqn_jaxtari_k4_hier_comb done.")
run_sh_command("CUDA_VISIBLE_DEVICES=12 uv run main.py +alg=pqn_jaxtari_k5_hier_comb_pre")
print("pqn_jaxtari_k5_hier_comb_pre done.")






