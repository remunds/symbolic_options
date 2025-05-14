import os
import time
import threading
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
import pygame
from jaxatari.renderers import AtraJaxisRenderer, PyGameRenderer
from jaxatari.wrappers import AtariState, MultiRewardLogEnvState as JaxtariMultiRewardLogEnvState
from symbolic_options.purejaxql.craftax_wrappers import MultiRewardLogEnvState as CraftaxMultiRewardLogEnvState
from craftax.craftax.renderer import render_craftax_pixels # for craftax
from craftax.craftax_classic.renderer import render_craftax_pixels as render_craftax_classic_pixels# for craftax_classic
from craftax.craftax.constants import (
    BLOCK_PIXEL_SIZE_HUMAN,
)

import wandb

class SidebarRenderer:
    def __init__(self, frame_shape, sidebar_width=50, font_size=15, text_color=(255, 255, 255), bg_color=(0, 0, 0), highlight_color=(50, 50, 200)):
        """
        frame_shape: (3, H, W)
        """
        _, self.h, self.w = frame_shape
        self.sidebar_width = sidebar_width
        self.text_color = text_color
        self.bg_color = bg_color
        self.highlight_color = highlight_color

        pygame.init()
        pygame.font.init()

        self.font = pygame.font.SysFont("Arial", font_size)
        self.surface = pygame.Surface((self.w + sidebar_width, self.h))
        self.left_surface = pygame.Surface((self.w, self.h))

    def render(self, frame, texts, active_cell):
        """
        Add sidebar with text to a single frame.

        frame: (3, H, W) RGB numpy array
        texts: list of tuple of strings
        """

        self.surface.fill(self.bg_color)

        # Draw the frame on the left
        pygame.surfarray.blit_array(self.left_surface, frame.transpose(2, 1, 0))
        self.surface.blit(self.left_surface, (0, 0))

        # Draw text on the right
        cell_height = self.h // len(texts)
        for i, text in enumerate(texts):
            y = i * cell_height
            if i == active_cell:
                pygame.draw.rect(
                    self.surface,
                    self.highlight_color,
                    pygame.Rect(self.w, y, self.sidebar_width, cell_height)
                )
            t0_surface = self.font.render(text[0], True, self.text_color)
            t1_surface = self.font.render(text[1], True, self.text_color)
            t0_x = self.w + (self.sidebar_width- t0_surface.get_width()) // 2
            t1_x = self.w + (self.sidebar_width- t1_surface.get_width()) // 2
            total_height = t0_surface.get_height() + t1_surface.get_height() + 4 # 4 pixels of padding 
            start_y = y + (cell_height - total_height) // 2
            t0_y = start_y
            t1_y = start_y + t0_surface.get_height() + 4

            self.surface.blit(t0_surface, (t0_x, t0_y))
            self.surface.blit(t1_surface, (t1_x, t1_y))

        # Convert back to NumPy
        # Note: pygame.surfarray.array3d returns (W, H, 3) but we need (3, H, W)
        output_frame = pygame.surfarray.array3d(self.surface).transpose(2, 1, 0)
        return output_frame

    def close(self):
        pygame.quit()

class CraftaxRenderer:
    @partial(jax.jit, static_argnums=(0,2))
    def render(self, craftax_state, block_pixel_size=BLOCK_PIXEL_SIZE_HUMAN):
        return render_craftax_pixels(craftax_state, block_pixel_size=block_pixel_size)

class CraftaxClassicRenderer(CraftaxRenderer):
    @partial(jax.jit, static_argnums=(0,2))
    def render(self, craftax_state, block_pixel_size=BLOCK_PIXEL_SIZE_HUMAN):
        return render_craftax_classic_pixels(craftax_state, block_pixel_size=block_pixel_size)


# video_thread = None

def video_callback(states, active_agents, combined_qs, dones, step, renderer, modif=False):
    # global video_thread

    if renderer is None:
        print("Renderer is None, skipping video generation")
        return

    # if video_thread is not None and video_thread.is_alive():
    #     print("Thread is still running, skipping video generation")
    #     return
    
    video_thread = threading.Thread(target=collect_video, args=(states, active_agents, combined_qs, dones, step, renderer, modif))
    video_thread.start()

def add_active_agent(screen, active_agent_num: int):
    font = pygame.font.Font(None, 50)
    text_surface = font.render(f"active agent: {active_agent_num}", True, (255, 255, 255)) 
    screen.blit(text_surface, (300, 300)) 

def collect_video(states, active_agents, combined_qs, dones, step, renderer, modif=False):
    print("Rendering video...")
    video_folder = f"{wandb.run.dir}/media/videos/"
    os.makedirs(video_folder, exist_ok=True)
    if isinstance(states, JaxtariMultiRewardLogEnvState) or isinstance(states, CraftaxMultiRewardLogEnvState):
        states = states.env_state
    if isinstance(states, AtariState):
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

    if active_agents is not None:
        sidebar_renderer = SidebarRenderer(frames[0].shape)
        # (N, 3, H, W + sidebar_width)
        new_frames = np.zeros((frames.shape[0], frames.shape[1], frames.shape[2], frames.shape[3] + sidebar_renderer.sidebar_width), dtype=np.uint8)
        for i in range(len(frames)):
            # add sidebar to each frame
            texts = [(f"Option {t_i}:", f"{t:.2f}") for t_i, t in enumerate(combined_qs[i])]
            new_frames[i] = sidebar_renderer.render(frames[i], texts, active_agents[i])
        sidebar_renderer.close()
        frames = new_frames

    fps = 30 #if not isinstance(renderer, CraftaxRenderer) else 30
    video = wandb.Video(frames, fps=fps, format="mp4")
    name = f"video_{step}"
    if modif:
        name = f"video_{step}_modif"
    wandb.log({name: video}, step=wandb.run.step)
    print("Video done.")