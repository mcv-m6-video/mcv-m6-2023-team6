import moviepy.editor as mp


def video2gif(video_path, gif_path, start_time, duration, fps):
    """Convert a video to gif.

    Args:
        video_path (str): Path to the video.
        gif_path (str): Path to the gif.
        start_time (float): Start time of the gif.
        duration (float): Duration of the gif.
        fps (int): Frames per second of the gif.
    """
    # get the video
    video = mp.VideoFileClip(video_path)
    # resize the video
    video = video.resize(0.5)
    # get the gif
    if duration is not None:
        gif = video.subclip(start_time, start_time + duration)
    else:
        gif = video.subclip(start_time)
    gif.speedx(1).to_gif(gif_path, fps=fps)


video2gif('/ghome/group03/dataset/vdo_tallat.mp4',
           '/ghome/group03/dataset/vdo_tallat.gif', 0, None, 8)

# video2gif('/ghome/group03/mcv-m6-2023-team6/week4/Results/Task1_2/maskflownet/flow_video.mp4',
#            '/ghome/group03/mcv-m6-2023-team6/week4/Results/Task1_2/maskflownet/flow_video.gif', 0, None, 8)


# resize gif
def resize_gif(gif_path, new_gif_path, start_time, duration, fps):
    """Resize a gif.

    Args:
        gif_path (str): Path to the gif.
        new_gif_path (str): Path to the resized gif.
        scale (float): Scale of the resized gif.
    """
    # get the gif
    gif = mp.VideoFileClip(gif_path)
    # resize the gif
    # gif = gif.resize(0.5)
    # save the resized gif
    if duration is not None:
        gif = gif.subclip(start_time, start_time + duration)
    else:
        gif = gif.subclip(start_time)
    gif.speedx(10).to_gif(new_gif_path, fps=fps)


# resize_gif('../../week2/pixelevolution_vid.gif', '../../week2/pixelevolution_resize.gif', 0, None, 5)
# resize_gif('/ghome/group03/mcv-m6-2023-team6/week3/Results/Video_IoU/iou.gif',
#             '/ghome/group03/mcv-m6-2023-team6/week3/Results/Video_IoU/iou_speed.gif', 0, None, 5)