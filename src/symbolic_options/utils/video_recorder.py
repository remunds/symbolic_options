import os
import time
import threading
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
import pygame
from jaxtari.renderers import AtraJaxisRenderer, PyGameRenderer
from jaxtari.wrappers import MultiRewardLogEnvState as JaxtariMultiRewardLogEnvState
from symbolic_options.purejaxql.craftax_wrappers import MultiRewardLogEnvState as CraftaxMultiRewardLogEnvState
from craftax.craftax.renderer import render_craftax_pixels
from craftax.craftax.constants import (
    BLOCK_PIXEL_SIZE_HUMAN,
)

import wandb

class CraftaxRenderer:
    @partial(jax.jit, static_argnums=(0,))
    def render(self, craftax_state):
        render_fn = jax.jit(render_craftax_pixels, static_argnums=(1,))
        return render_fn(craftax_state, block_pixel_size=BLOCK_PIXEL_SIZE_HUMAN)


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
    text_surface = font.render(f"active agent: {active_agent_num}", True, (255, 255, 255)) 
    screen.blit(text_surface, (300, 300)) 

def collect_video(states, active_agents, dones, step, renderer):
    print("Rendering video...")
    video_folder = f"{wandb.run.dir}/media/videos/"
    os.makedirs(video_folder, exist_ok=True)
    if isinstance(states, JaxtariMultiRewardLogEnvState) or isinstance(states, CraftaxMultiRewardLogEnvState):
        states = states.env_state

    # num_states is where the first done is True
    num_states = jnp.argmax(dones)
    # or len of the first array of the states pytree
    if num_states == 0:
        num_states = len(states[0])

    # select every 4th frame (and only the first num_states)
    # states_reduced = jax.tree_util.tree_map(lambda x: x[:num_states][::4], states)
    states_reduced = jax.tree_util.tree_map(lambda x: x[:num_states], states)
    if isinstance(renderer, AtraJaxisRenderer) or isinstance(renderer, CraftaxRenderer):
        rasters = jax.vmap(renderer.render)(states_reduced)
        frames = np.array(rasters, dtype=np.uint8)
    elif isinstance(renderer, PyGameRenderer):
        pygame.init()
        # select every 4th frame (and only the first num_states)
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

    # for jaxtari
    if isinstance(renderer, AtraJaxisRenderer) or isinstance(renderer, PyGameRenderer):
        # shape currently is (N, W, H, 3)
        # but should be (N, 3, H, W)
        frames = np.transpose(frames, (0, 3, 2, 1))
    else: # for craftax
        # shape currently is (N, H, W, 3)
        # but should be (N, 3, H, W) 
        frames = np.transpose(frames, (0, 3, 1, 2))

    video = wandb.Video(frames, fps=60, format="mp4")
    wandb.log({f"video_{step}": video}, step=wandb.run.step)
    print("Video done.")