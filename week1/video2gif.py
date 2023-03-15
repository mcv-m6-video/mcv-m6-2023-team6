from moviepy.video.io.VideoFileClip import VideoFileClip


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
    video = VideoFileClip(video_path)
    # get the gif
    gif = video.subclip(start_time, start_time + duration)
    gif.write_gif(gif_path, fps=fps)


video2gif('./runs/noisy/video.mp4', './runs/noisy/video.gif', 0, 5, 10)