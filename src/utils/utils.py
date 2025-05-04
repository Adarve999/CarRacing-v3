# Set up fake display; otherwise rendering will fail
import gymnasium as gym
from gymnasium.wrappers import (RecordVideo)
import base64
from pathlib import Path
from IPython import display as ipythondisplay

def make_env(continuousMode = False):
    def _init():
        env = gym.make("CarRacing-v3", render_mode=None, continuous= continuousMode)
        return env
    return _init 


def make_eval_env(modelIdentifier="model", max_steps=2000 ,record=False,continuousMode = False):
    env = gym.make(
        "CarRacing-v3",
        render_mode="rgb_array",
        continuous=continuousMode,
        max_episode_steps=max_steps)

    if record:
        env = RecordVideo(
            env,
            video_folder=f"../videos/{modelIdentifier}/",
            name_prefix=f"{modelIdentifier}",
            episode_trigger=lambda ep: True,
            video_length=0
        )
    return env


def show_videos(video_path="", prefix=""):
    """
    Taken from https://github.com/eleurent/highway-env

    :param video_path: (str) Path to the folder containing videos
    :param prefix: (str) Filter the video, showing only the only starting with this prefix
    """
    html = []
    for mp4 in Path(video_path).glob("{}*.mp4".format(prefix)):
        video_b64 = base64.b64encode(mp4.read_bytes())
        html.append(
            """<video alt="{}" autoplay 
                    loop controls style="height: 400px;">
                    <source src="data:video/mp4;base64,{}" type="video/mp4" />
                </video>""".format(
                mp4, video_b64.decode("ascii")
            )
        )
    ipythondisplay.display(ipythondisplay.HTML(data="<br>".join(html)))