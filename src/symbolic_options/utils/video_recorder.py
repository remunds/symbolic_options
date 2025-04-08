import os
import threading

import jax
import jax.numpy as jnp
import numpy as np
import pygame
from jaxtari.renderers import AtraJaxisRenderer, PyGameRenderer
from jaxtari.wrappers import MultiRewardLogEnvState

import wandb

video_thread = None

def video_callback(states, active_agents, dones, step, renderer):
    global video_thread
    if renderer is None:
        print("Renderer is None, skipping video generation")
        return

    if video_thread is not None and video_thread.is_alive():
        print("Thread is still running, skipping video generation")
        return
    
    video_thread = threading.Thread(target=collect_video, args=(states, active_agents, dones, step, renderer))
    video_thread.start()

def add_active_agent(screen, active_agent_num: int):
    font = pygame.font.Font(None, 50)
    # text_surface = font.render(str(active_agent_num), True, (255, 255, 255)) 
    text_surface = font.render(f"active agent: {active_agent_num}", True, (255, 255, 255)) 
    screen.blit(text_surface, (300, 300)) 

def collect_video(states, active_agents, dones, step, renderer):

    video_folder = f"{wandb.run.dir}/media/videos/"
    os.makedirs(video_folder, exist_ok=True)
    if isinstance(states, MultiRewardLogEnvState):
        states = states.env_state

    # num_states is where the first done is True
    num_states = jnp.argmax(dones)
    # or len of the first array of the states pytree
    if num_states == 0:
        num_states = len(states[0])

    if isinstance(renderer, AtraJaxisRenderer):
        rasters = jax.vmap(renderer.render)(states)
        # select every 4th frame (and only the first num_states)
        frames = np.array(rasters[:num_states][::4],dtype=np.uint8)
    elif isinstance(renderer, PyGameRenderer):
        pygame.init()
        # select every 4th frame (and only the first num_states)
        states_reduced = jax.tree_util.tree_map(lambda x: x[:num_states][::4], states)
        reduced_state_num = jax.tree_util.tree_leaves(states_reduced)[0].shape[0]
        frames = [] 
        for i in range(reduced_state_num):
            # select i'th frame of every state
            states_i = jax.tree_util.tree_map(lambda x: x[i], states_reduced)
            renderer.render(states_i)
            if active_agents is not None:
                add_active_agent(renderer.screen, active_agents[i*4])
            frame = pygame.surfarray.array3d(renderer.screen)
            frames.append(frame)

        # convert to numpy array
        frames = np.array(frames, dtype=np.uint8)

    else:
        print("Renderer is not known, skipping video generation")
        return
    # shape currently is (N, H, W, 3)
    # but should be (N, 3, H, W)
    frames = np.transpose(frames, (0, 3, 2, 1))
    video = wandb.Video(frames, fps=64, format="mp4")
    wandb.log({f"video_{step}": video}, step=wandb.run.step)