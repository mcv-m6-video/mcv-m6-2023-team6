import sys
sys.path.insert(0,'week3')

import os

def createFolders(seq_names,seq_lengths,gts_folder):

    # Create folders GT
    if not os.path.exists('TrackEval/data/gt/mot_challenge/MOT17-train'):
        os.makedirs('TrackEval/data/gt/mot_challenge/MOT17-train')
    if not os.path.exists('TrackEval/data/gt/mot_challenge/seqmaps'):
        os.makedirs('TrackEval/data/gt/mot_challenge/seqmaps')
    with open('TrackEval/data/gt/mot_challenge/seqmaps/MOT17-train.txt', "w+") as f3:
            f3.write('name\n')
    # Create folders trackers
    if not os.path.exists('TrackEval/data/trackers/mot_challenge/MOT17-train/MPNTrack/data'):
        os.makedirs('TrackEval/data/trackers/mot_challenge/MOT17-train/MPNTrack/data')
    
    
    for seq,i in enumerate(seq_names):
        if not os.path.exists('TrackEval/data/gt/mot_challenge/MOT17-train/'+seq):
            os.makedirs('TrackEval/data/gt/mot_challenge/MOT17-train/'+seq)
        if not os.path.exists('TrackEval/data/gt/mot_challenge/MOT17-train/'+seq+'/gt'):
            os.makedirs('TrackEval/data/gt/mot_challenge/MOT17-train/'+seq+'/gt')

        #open the gt file (named seq ) and save it to another txt file
        with open(os.path.join(gts_folder, seq + ".txt"), "r") as f:
            with open('TrackEval/data/gt/mot_challenge/MOT17-train/'+seq+'/gt/gt.txt', "w+") as f1:
                for line in f:
                    line = line.split(',')
                    line[0] = str(int(line[0]) + 1)
                    line = ','.join(line)
                    f1.write(line)
         
        with open('TrackEval/data/gt/mot_challenge/MOT17-train/'+seq+'/seqinfo.ini', "w+") as f2:
            f2.write('[Sequence]\n')
            f2.write('name = '+seq + '\n')
            f2.write('imDir = img1\n')
            f2.write('frameRate = 10\n')
            f2.write('seqLength = '+str(seq_lengths[i])+'\n')
            f2.write('imWidth = 1920\n')
            f2.write('imHeight = 1080\n')
            f2.write('imExt = .jpg\n')

        #append the name of the sequence to the seqmaps file
        with open('TrackEval/data/gt/mot_challenge/seqmaps/MOT17-train.txt', "a") as f3:
            f3.write(seq + '\n')
        
if __name__ == "__main__":
    seq_names = ['c010', 'c011', 'c012', 'c013', 'c014', 'c015']
    seq_lengths = [2141, 2141, 2141, 2141, 2141, 2141]
    gts_folder = 'GT/'
    createFolders(seq_names,seq_lengths, gts_folder)