import cv2
from matplotlib import pyplot as plt


# Rendering Video AICity Challenge 2023
def rendering_video(cfg, gt_boxes=None):
    """
    :param path:
    :param visualize:
    :return:
    """
    iou_list = []
    # TODO: Implement the rendering of the video WIP
    cap = cv2.VideoCapture(cfg.video_path)
    frame_cont = 0
    ret, frame = cap.read()
    wait_time = 1
    while ret:
        frame_cont += 1
        if cfg.display:
            cv2.imshow('frame', frame)
            k = cv2.waitKey(wait_time)
            if k == ord('q'):
                break
            elif k == ord('s'):
                cv2.imwrite(f'save_{frame_cont}.png', frame)
            elif k == ord('p'):
                wait_time = int(not (bool(wait_time)))
        ret, frame = cap.read()

    if cfg.save:
        plt.figure()
        plt.plot(iou_list)
        plt.xlim([0, 2140])
        plt.ylim([0, 1])
        plt.savefig(f'runs/{cfg.run_name}/iou_plt_{frame_cont}.png')

        cv2.imwrite(f'runs/{cfg.run_name}/frame_{frame_cont}.png', frame)
