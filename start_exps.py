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

run_sh_command("CUDA_VISIBLE_DEVICES=14 uv run main.py +alg=pqn_jaxtari_sea4_hier_comb")
print("pqn_jaxtari_sea4_hier_comb done.")






