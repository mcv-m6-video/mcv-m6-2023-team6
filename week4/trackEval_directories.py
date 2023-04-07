import sys
sys.path.insert(0,'week3')

import os

def createFolders(seq_names,seq,seq_lengths,gts_folder):

    # Create folders GT
    if not os.path.exists('TrackEval_'+seq+'/data/gt/mot_challenge/MOT17-train'):
        os.makedirs('TrackEval_'+seq+'/data/gt/mot_challenge/MOT17-train')
    if not os.path.exists('TrackEval_'+seq+'/data/gt/mot_challenge/seqmaps'):
        os.makedirs('TrackEval_'+seq+'/data/gt/mot_challenge/seqmaps')
    with open('TrackEval_'+seq+'/data/gt/mot_challenge/seqmaps/MOT17-train.txt', "w+") as f3:
            f3.write('name\n')
    # Create folders trackers
    if not os.path.exists('TrackEval_'+seq+'/data/trackers/mot_challenge/MOT17-train/MPNTrack/data'):
        os.makedirs('TrackEval_'+seq+'/data/trackers/mot_challenge/MOT17-train/MPNTrack/data')
    
    
    for seq,i in enumerate(seq_names):
        if not os.path.exists('TrackEval_'+seq+'/data/gt/mot_challenge/MOT17-train/'+seq):
            os.makedirs('TrackEval_'+seq+'/data/gt/mot_challenge/MOT17-train/'+seq)
        if not os.path.exists('TrackEval_'+seq+'/data/gt/mot_challenge/MOT17-train/'+seq+'/gt'):
            os.makedirs('TrackEval_'+seq+'/data/gt/mot_challenge/MOT17-train/'+seq+'/gt')

        #open the gt file (named seq ) and save it to another txt file
        with open(os.path.join(gts_folder, seq + ".txt"), "r") as f:
            with open('TrackEval_'+seq+'/data/gt/mot_challenge/MOT17-train/'+seq+'/gt/gt.txt', "w+") as f1:
                for line in f:
                    line = line.split(',')
                    line[0] = str(int(line[0]) + 1)
                    line = ','.join(line)
                    f1.write(line)
         
        with open('TrackEval_'+seq+'/data/gt/mot_challenge/MOT17-train/'+seq+'/seqinfo.ini', "w+") as f2:
            f2.write('[Sequence]\n')
            f2.write('name = '+seq + '\n')
            f2.write('imDir = img1\n')
            if(seq == 'c015'):
                f2.write('frameRate = 8\n')
            else:
                f2.write('frameRate = 10\n')
            f2.write('seqLength = '+str(seq_lengths[i])+'\n')
            f2.write('imWidth = 1920\n')
            f2.write('imHeight = 1080\n')
            f2.write('imExt = .jpg\n')

        #append the name of the sequence to the seqmaps file
        with open('TrackEval_'+seq+'/data/gt/mot_challenge/seqmaps/MOT17-train.txt', "a") as f3:
            f3.write(seq + '\n')
        
if __name__ == "__main__":

    #create TrackEval for task 1.3 
    ###### FALTA
    seq_names = ['unknown']
    seq_lengths = [0]
    gts_folder = 'GT/KITTY/'
    createFolders(seq_names,'kitty',seq_lengths, gts_folder)


    # Create TrackEval for test task 2
    seq_names_test = ['c010', 'c011', 'c012', 'c013', 'c014', 'c015']
    seq_lengths_test = [2141, 2279, 2422, 2415, 2332, 1928]
    gts_folder_test = 'GT/AI_test/'
    createFolders(seq_names_test,'test',seq_lengths_test, gts_folder_test)

    # Create TrackEval for train task 2
    seq_names_train = ['c001', 'c002', 'c003', 'c004', 'c005', 'c016', 'c017', 'c018', 'c019', 'c020','c021', 'c022', 'c023', 'c024', 'c025', 'c026', 'c027', 'c028', 'c029', 'c030', 'c031', 'c032',  'c033', 'c034', 'c035','c031', 'c036',  'c037', 'c038', 'c039','c040']
    seq_lengths_train = []
    gts_folder_train = 'GT/AI_train/'
    createFolders(seq_names_train,'train',seq_lengths_train, gts_folder_train)