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
# run_sh_command("CUDA_VISIBLE_DEVICES=6 uv run main.py +alg=pqn_jaxtari_sea2_hier_baseline")
# print("sea2 baseline done.")
# run_sh_command("CUDA_VISIBLE_DEVICES=6 uv run main.py +alg=pqn_jaxtari_sea2_hier")
# print("sea2 done.")
# # run_sh_command("CUDA_VISIBLE_DEVICES=15 uv run main.py +alg=pqn_jaxtari_sea3_hier_llm")
# # print("sea3 done.")
# run_sh_command("CUDA_VISIBLE_DEVICES=6 uv run main.py +alg=pqn_jaxtari_sea4_hier_comb")
# print("sea4 done.")

# same for k1 - k4
# run_sh_command("CUDA_VISIBLE_DEVICES=14 uv run main.py +alg=pqn_jaxtari_k1")
# print("k1 done.")
# run_sh_command("CUDA_VISIBLE_DEVICES=7 uv run main.py +alg=pqn_jaxtari_k2_hier_baseline")
# print("k2 baseline done.")
# run_sh_command("CUDA_VISIBLE_DEVICES=7 uv run main.py +alg=pqn_jaxtari_k2_hier")
# print("k2 done.")
# # run_sh_command("CUDA_VISIBLE_DEVICES=14 uv run main.py +alg=pqn_jaxtari_k3_hier_llm")
# # print("k3 done.")
# run_sh_command("CUDA_VISIBLE_DEVICES=7 uv run main.py +alg=pqn_jaxtari_k4_hier_comb")
# print("k4 done.")   

# # same for pong1 - pong4
# run_sh_command("CUDA_VISIBLE_DEVICES=13 uv run main.py +alg=pqn_jaxtari_pong1")
# print("pong1 done.")
# run_sh_command("CUDA_VISIBLE_DEVICES=8 uv run main.py +alg=pqn_jaxtari_pong2_hier_baseline")
# print("pong2 baseline done.")
# run_sh_command("CUDA_VISIBLE_DEVICES=8 uv run main.py +alg=pqn_jaxtari_pong2_hier")
# print("pong2 done.")
# # run_sh_command("CUDA_VISIBLE_DEVICES=13 uv run main.py +alg=pqn_jaxtari_pong3_hier_llm")
# # print("pong3 done.")
# run_sh_command("CUDA_VISIBLE_DEVICES=8 uv run main.py +alg=pqn_jaxtari_pong4_hier_comb")
# print("pong4 done.")

# same for breakout1 - breakout4
# # run_sh_command("CUDA_VISIBLE_DEVICES=12 uv run main.py +alg=pqn_jaxtari_breakout1")
# print("breakout1 done.")
# run_sh_command("CUDA_VISIBLE_DEVICES=9 uv run main.py +alg=pqn_jaxtari_breakout2_hier_baseline")
# print("breakout2 baseline done.")
# run_sh_command("CUDA_VISIBLE_DEVICES=9 uv run main.py +alg=pqn_jaxtari_breakout2_hier")
# print("breakout2 done.")
# # run_sh_command("CUDA_VISIBLE_DEVICES=12 uv run main.py +alg=pqn_jaxtari_breakout3_hier_llm")
# # print("breakout3 done.")
# run_sh_command("CUDA_VISIBLE_DEVICES=9 uv run main.py +alg=pqn_jaxtari_breakout4_hier_comb")
# print("breakout4 done.")

# # same for freeway1 - freeway4
# run_sh_command("CUDA_VISIBLE_DEVICES=11 uv run main.py +alg=pqn_jaxtari_freeway1")
# print("freeway1 done.")
# run_sh_command("CUDA_VISIBLE_DEVICES=10 uv run main.py +alg=pqn_jaxtari_freeway2_hier_baseline")
# print("freeway2 baseline done.")
# run_sh_command("CUDA_VISIBLE_DEVICES=10 uv run main.py +alg=pqn_jaxtari_freeway2_hier")
# print("freeway2 done.")
# # run_sh_command("CUDA_VISIBLE_DEVICES=11 uv run main.py +alg=pqn_jaxtari_freeway3_hier_llm")
# # print("freeway3 done.")
# run_sh_command("CUDA_VISIBLE_DEVICES=10 uv run main.py +alg=pqn_jaxtari_freeway4_hier_comb")
# print("freeway4 done.") 

# PPO's
# run_sh_command("CUDA_VISIBLE_DEVICES=10 uv run src/symbolic_options/ppo_jaxtari.py +alg=ppo_jaxtari_pong")
# print("pong done.")
# run_sh_command("CUDA_VISIBLE_DEVICES=9 uv run src/symbolic_options/ppo_jaxtari.py +alg=ppo_jaxtari_freeway")
# print("freeway done.")
# run_sh_command("CUDA_VISIBLE_DEVICES=8 uv run src/symbolic_options/ppo_jaxtari.py +alg=ppo_jaxtari_breakout")
# print("breakout done.")
# run_sh_command("CUDA_VISIBLE_DEVICES=7 uv run src/symbolic_options/ppo_jaxtari.py +alg=ppo_jaxtari_seaquest")
# print("seaquest done.")
# run_sh_command("CUDA_VISIBLE_DEVICES=0 uv run src/symbolic_options/ppo_jaxtari.py +alg=ppo_jaxtari_kangaroo")
# print("kangaroo done.") 


# Craftax RNN (3 seeds each)
# run_sh_command("CUDA_VISIBLE_DEVICES=15 uv run main.py +alg=pqn_craftax_rnn SEED=0")
# print("craftax rnn 1 done.")
# run_sh_command("CUDA_VISIBLE_DEVICES=15 uv run main.py +alg=pqn_craftax_rnn SEED=1")
# print("craftax rnn 2 done.")
# run_sh_command("CUDA_VISIBLE_DEVICES=15 uv run main.py +alg=pqn_craftax_rnn SEED=2")
# print("craftax rnn 3 done.")

# run_sh_command("CUDA_VISIBLE_DEVICES=14 uv run main.py +alg=pqn_craftax_rnn_hier_baseline SEED=0")
# print("craftax rnn hier baseline 1 done.")
# run_sh_command("CUDA_VISIBLE_DEVICES=14 uv run main.py +alg=pqn_craftax_rnn_hier_baseline SEED=1")
# print("craftax rnn hier baseline 2 done.")
# run_sh_command("CUDA_VISIBLE_DEVICES=14 uv run main.py +alg=pqn_craftax_rnn_hier_baseline SEED=2")
# print("craftax rnn hier baseline 3 done.")

# run_sh_command("CUDA_VISIBLE_DEVICES=13 uv run main.py +alg=pqn_craftax_rnn_hier SEED=0")
# print("craftax rnn hier 1 done.")
# run_sh_command("CUDA_VISIBLE_DEVICES=13 uv run main.py +alg=pqn_craftax_rnn_hier SEED=1")
# print("craftax rnn hier 2 done.")
# run_sh_command("CUDA_VISIBLE_DEVICES=13 uv run main.py +alg=pqn_craftax_rnn_hier SEED=2")
# print("craftax rnn hier 3 done.")

# run_sh_command("CUDA_VISIBLE_DEVICES=12 uv run main.py +alg=pqn_craftax_rnn_hier_llm SEED=0")
# print("craftax rnn hier llm 1 done.")
# run_sh_command("CUDA_VISIBLE_DEVICES=12 uv run main.py +alg=pqn_craftax_rnn_hier_llm SEED=1")
# print("craftax rnn hier llm 2 done.")
# run_sh_command("CUDA_VISIBLE_DEVICES=12 uv run main.py +alg=pqn_craftax_rnn_hier_llm SEED=2")
# print("craftax rnn hier llm 3 done.")

# run_sh_command("CUDA_VISIBLE_DEVICES=11 uv run main.py +alg=pqn_craftax_rnn_hier_comb SEED=0")
# print("craftax rnn hier comb 1 done.")
# run_sh_command("CUDA_VISIBLE_DEVICES=11 uv run main.py +alg=pqn_craftax_rnn_hier_comb SEED=1")
# print("craftax rnn hier comb 2 done.")
# run_sh_command("CUDA_VISIBLE_DEVICES=11 uv run main.py +alg=pqn_craftax_rnn_hier_comb SEED=2")
# print("craftax rnn hier comb 3 done.")

# ---- Noisy experiments ----
# run_sh_command("CUDA_VISIBLE_DEVICES=0 uv run main.py +alg=pqn_jaxtari_sea2_hier_noisy0")
# run_sh_command("CUDA_VISIBLE_DEVICES=0 uv run main.py +alg=pqn_jaxtari_sea2_hier_noisy1")
# print("sea2 done.")
# run_sh_command("CUDA_VISIBLE_DEVICES=2 uv run main.py +alg=pqn_jaxtari_k2_hier_noisy0")
# run_sh_command("CUDA_VISIBLE_DEVICES=2 uv run main.py +alg=pqn_jaxtari_k2_hier_noisy1")
# print("k2 done.")

# run_sh_command("CUDA_VISIBLE_DEVICES=2 uv run main.py +alg=pqn_jaxtari_sea3_hier_llm_noisy0")
# run_sh_command("CUDA_VISIBLE_DEVICES=2 uv run main.py +alg=pqn_jaxtari_sea3_hier_llm_noisy1")
# print("sea3 done.")
# run_sh_command("CUDA_VISIBLE_DEVICES=2 uv run main.py +alg=pqn_jaxtari_k3_hier_llm_noisy0")
# run_sh_command("CUDA_VISIBLE_DEVICES=2 uv run main.py +alg=pqn_jaxtari_k3_hier_llm_noisy1")
# print("k3 done.")

# run_sh_command("CUDA_VISIBLE_DEVICES=3 uv run main.py +alg=pqn_jaxtari_sea4_hier_comb_noisy0")
# run_sh_command("CUDA_VISIBLE_DEVICES=3 uv run main.py +alg=pqn_jaxtari_sea4_hier_comb_noisy1")
# print("sea4 done.")
# run_sh_command("CUDA_VISIBLE_DEVICES=2 uv run main.py +alg=pqn_jaxtari_k4_hier_comb_noisy0")
# run_sh_command("CUDA_VISIBLE_DEVICES=2 uv run main.py +alg=pqn_jaxtari_k4_hier_comb_noisy1")
# print("k4 done.")


# run_sh_command("CUDA_VISIBLE_DEVICES=3 uv run main.py +alg=pqn_jaxtari_sea2_hier_noisy2")
# run_sh_command("CUDA_VISIBLE_DEVICES=3 uv run main.py +alg=pqn_jaxtari_sea2_hier_noisy3")
# print("sea2 done.")
# run_sh_command("CUDA_VISIBLE_DEVICES=3 uv run main.py +alg=pqn_jaxtari_k2_hier_noisy2")
# run_sh_command("CUDA_VISIBLE_DEVICES=3 uv run main.py +alg=pqn_jaxtari_k2_hier_noisy3")
# print("k2 done.")

# run_sh_command("CUDA_VISIBLE_DEVICES=0 uv run main.py +alg=pqn_jaxtari_sea3_hier_llm_noisy2")
# run_sh_command("CUDA_VISIBLE_DEVICES=0 uv run main.py +alg=pqn_jaxtari_sea3_hier_llm_noisy3")
# print("sea3 done.")
# run_sh_command("CUDA_VISIBLE_DEVICES=0 uv run main.py +alg=pqn_jaxtari_k3_hier_llm_noisy2")
# run_sh_command("CUDA_VISIBLE_DEVICES=0 uv run main.py +alg=pqn_jaxtari_k3_hier_llm_noisy3")
# print("k3 done.")

run_sh_command("CUDA_VISIBLE_DEVICES=5 uv run main.py +alg=pqn_jaxtari_sea4_hier_comb_noisy2")
run_sh_command("CUDA_VISIBLE_DEVICES=5 uv run main.py +alg=pqn_jaxtari_sea4_hier_comb_noisy3")
print("sea4 done.")
run_sh_command("CUDA_VISIBLE_DEVICES=5 uv run main.py +alg=pqn_jaxtari_k4_hier_comb_noisy2")
run_sh_command("CUDA_VISIBLE_DEVICES=5 uv run main.py +alg=pqn_jaxtari_k4_hier_comb_noisy3")
print("k4 done.")