import matplotlib.pyplot as plt
import numpy as np
import cv2
import os
import imageio





def plot_iou_score(frames_id, frames_num, iou_scores, video_capture, gt_boxes, predicted_boxes_group, fps, display=False):
    mages_plot = []
    fig = plt.figure(figsize=(5, 5))
    # Set the title
    fig.suptitle('IoU score for each frame')
    fig.tight_layout(pad=0)
    ax = plt.axes()
    # Set the x label
    ax.set_xlabel('Frame')
    # Set the y label
    ax.set_ylabel('IoU score')
    # Set the x axis range
    ax.set_xlim(0, frames_num[-1])
    # Set the y axis range
    ax.set_ylim(0, 1)
    # Create a line
    (line,) = ax.plot([], [], lw=2)
    line.set_data([], [])
    if display:
        fig.show()
    fig.canvas.draw()

    # Loop through each frame
    for i, frame_id in enumerate(frames_id):
        # Read the frame
        ret, frame = video_capture.read()
        if ret:
            # Convert the frame to RGB
            # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # Draw the ground truth boxes
            for box in gt_boxes[frame_id]:
                cv2.rectangle(frame, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (0, 255, 0), 2)
            # Draw the predicted boxes
            for box in predicted_boxes_group[frame_id]:
                cv2.rectangle(frame, (int(box[1]), int(box[2])), (int(box[3]), int(box[4])), (0, 0, 255), 2)
            # Draw the IoU score
            iou_score = round(iou_scores[i], 2)
            cv2.putText(
                frame, f"IoU score: {iou_score}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA
            )
            # put text number of frame
            cv2.putText(
                frame, f"Frame: {frames_num[i]}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA
            )
            # put fps
            cv2.putText(frame, f"FPS: {fps}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
            # Write the frame to the video
            out.write(frame)
            line.set_data(frames_num[:i], iou_scores[:i])
            fig.canvas.draw()
            image = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
            image = image.reshape(fig.canvas.get_width_height(physical=True)[::-1] + (3,))

            images_plot.append(image)
            if display:
                cv2.imshow('frame', frame)
                k = cv2.waitKey(wait_time)
                if k == ord('q'):
                    break
                elif k == ord('s'):
                    cv2.imwrite(f'save_{frames_num[i]}.png', frame)
                elif k == ord('p'):
                    wait_time = int(not (bool(wait_time)))
    print("mAP: ", AP)
    print("mIOU: ", mIOU)
    time_end = time.time()
    print("Time: ", time_end - time_start)
    if save:
        # Release the VideoWriter object
        out.release()
        # create gif matplotlib figure
        # !convert -delay 10 -loop 0 *.png animation.gif
        imageio.mimsave(path + 'iou.gif', images_plot)
    time_end = time.time()
    print("Time_Finished: ", time_end - time_start)