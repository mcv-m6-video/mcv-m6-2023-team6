import pandas as pd


def save_txt(tracking_boxes,out_path,sufix=None)
  # save trackers to data frame, not out
  # save to csv
  df_list = []
  for frame_id in tracking_boxes:
      for track in tracking_boxes[frame_id]:
          width = track[3] - track[1]
          height = track[4] - track[2]
          bb_left = track[1]
          bb_top = track[2]
          df_list.append(pd.DataFrame({'frame': int(frame_id), 'id': track[-1], 'bb_left': bb_left, 'bb_top': bb_top,
                                       'bb_width': width, 'bb_height': height, 'conf': track[-2], "x": -1, "y": -1,
                                       "z": -1}, index=[0]))
  #format output for the evaluation 
  df = pd.concat(df_list, ignore_index=True)
  df = df.sort_values(by=['id'])
  df['frame'] = df['frame'] + 1
  
  
  if sufix:
    with open(f'{out_path}/MOT_17_22_{sufix}.txt', 'a') as f:
        text = df.to_string(header=False, index=False)
        f.write(text)
        
  else:
    with open(f'{out_path}/MOT_17_22_{sufix}.txt', 'a') as f:
        text = df.to_string(header=False, index=False)
        f.write(text)