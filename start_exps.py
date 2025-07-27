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

# run_sh_command("CUDA_VISIBLE_DEVICES=15 uv run main.py +alg=pqn_jaxtari_sea1")
# print("sea1 done.")
# run_sh_command("CUDA_VISIBLE_DEVICES=15 uv run main.py +alg=pqn_jaxtari_sea2_hier_baseline")
# print("sea2 baseline done.")
# run_sh_command("CUDA_VISIBLE_DEVICES=15 uv run main.py +alg=pqn_jaxtari_sea2_hier")
# print("sea2 done.")
# run_sh_command("CUDA_VISIBLE_DEVICES=15 uv run main.py +alg=pqn_jaxtari_sea3_hier_llm")
# print("sea3 done.")
# run_sh_command("CUDA_VISIBLE_DEVICES=15 uv run main.py +alg=pqn_jaxtari_sea4_hier_comb")
# print("sea4 done.")

# same for k1 - k4
# run_sh_command("CUDA_VISIBLE_DEVICES=14 uv run main.py +alg=pqn_jaxtari_k1")
# print("k1 done.")
# run_sh_command("CUDA_VISIBLE_DEVICES=14 uv run main.py +alg=pqn_jaxtari_k2_hier_baseline")
# print("k2 baseline done.")
# run_sh_command("CUDA_VISIBLE_DEVICES=14 uv run main.py +alg=pqn_jaxtari_k2_hier")
# print("k2 done.")
# run_sh_command("CUDA_VISIBLE_DEVICES=14 uv run main.py +alg=pqn_jaxtari_k3_hier_llm")
# print("k3 done.")
# run_sh_command("CUDA_VISIBLE_DEVICES=14 uv run main.py +alg=pqn_jaxtari_k4_hier_comb")
# print("k4 done.")   

# same for pong1 - pong4
# run_sh_command("CUDA_VISIBLE_DEVICES=13 uv run main.py +alg=pqn_jaxtari_pong1")
# print("pong1 done.")
# run_sh_command("CUDA_VISIBLE_DEVICES=13 uv run main.py +alg=pqn_jaxtari_pong2_hier_baseline")
# print("pong2 baseline done.")
# run_sh_command("CUDA_VISIBLE_DEVICES=13 uv run main.py +alg=pqn_jaxtari_pong2_hier")
# print("pong2 done.")
# run_sh_command("CUDA_VISIBLE_DEVICES=13 uv run main.py +alg=pqn_jaxtari_pong3_hier_llm")
# print("pong3 done.")
# run_sh_command("CUDA_VISIBLE_DEVICES=13 uv run main.py +alg=pqn_jaxtari_pong4_hier_comb")
# print("pong4 done.")

# same for breakout1 - breakout4
# run_sh_command("CUDA_VISIBLE_DEVICES=12 uv run main.py +alg=pqn_jaxtari_breakout1")
# print("breakout1 done.")
# run_sh_command("CUDA_VISIBLE_DEVICES=12 uv run main.py +alg=pqn_jaxtari_breakout2_hier_baseline")
# print("breakout2 baseline done.")
# run_sh_command("CUDA_VISIBLE_DEVICES=12 uv run main.py +alg=pqn_jaxtari_breakout2_hier")
# print("breakout2 done.")
# run_sh_command("CUDA_VISIBLE_DEVICES=12 uv run main.py +alg=pqn_jaxtari_breakout3_hier_llm")
# print("breakout3 done.")
# run_sh_command("CUDA_VISIBLE_DEVICES=12 uv run main.py +alg=pqn_jaxtari_breakout4_hier_comb")
# print("breakout4 done.")

# same for freeway1 - freeway4
# run_sh_command("CUDA_VISIBLE_DEVICES=11 uv run main.py +alg=pqn_jaxtari_freeway1")
# print("freeway1 done.")
# run_sh_command("CUDA_VISIBLE_DEVICES=11 uv run main.py +alg=pqn_jaxtari_freeway2_hier_baseline")
# print("freeway2 baseline done.")
# run_sh_command("CUDA_VISIBLE_DEVICES=11 uv run main.py +alg=pqn_jaxtari_freeway2_hier")
# print("freeway2 done.")
# run_sh_command("CUDA_VISIBLE_DEVICES=11 uv run main.py +alg=pqn_jaxtari_freeway3_hier_llm")
# print("freeway3 done.")
# run_sh_command("CUDA_VISIBLE_DEVICES=11 uv run main.py +alg=pqn_jaxtari_freeway4_hier_comb")
# print("freeway4 done.") 

# PPO's
run_sh_command("CUDA_VISIBLE_DEVICES=10 uv run src/symbolic_options/ppo_jaxtari.py +alg=ppo_jaxtari_pong")
print("pong done.")
run_sh_command("CUDA_VISIBLE_DEVICES=10 uv run src/symbolic_options/ppo_jaxtari.py +alg=ppo_jaxtari_freeway")
print("freeway done.")
run_sh_command("CUDA_VISIBLE_DEVICES=10 uv run src/symbolic_options/ppo_jaxtari.py +alg=ppo_jaxtari_breakout")
print("breakout done.")
run_sh_command("CUDA_VISIBLE_DEVICES=10 uv run src/symbolic_options/ppo_jaxtari.py +alg=ppo_jaxtari_seaquest")
print("seaquest done.")
run_sh_command("CUDA_VISIBLE_DEVICES=10 uv run src/symbolic_options/ppo_jaxtari.py +alg=ppo_jaxtari_kangaroo")
print("kangaroo done.") 