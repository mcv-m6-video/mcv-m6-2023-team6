import os

path = "/ghome/group03/mcv-m6-2023-team6/week5/Results/end2end/byTrack"
outputPath = "/ghome/group03/mcv-m6-2023-team6/week5/TrackEval/S01/data/trackers/mot_challenge/MOT17-train/MPNTrack/data"

endToEnd = True
MTMC = False
seq = "S03"
cams = {"S01":['c001', 'c002', 'c003', 'c004', 'c005'], "S03": ['c010', 'c011', 'c012', 'c013', 'c014', 'c015'], "S04": ['c016', 'c017', 'c018', 'c019', 'c020','c021', 'c022', 'c023', 'c024', 'c025', 'c026', 'c027', 'c028', 'c029', 'c030', 'c031', 'c032',  'c033', 'c034', 'c035','c036',  'c037', 'c038', 'c039','c040']}

if endToEnd:
    
    for folder in os.listdir(path):
        #iterate the subfolders (path + folder, only if it is a folder)
        if os.path.isdir(os.path.join(path, folder)):
            for subfolder in os.listdir(os.path.join(path, folder)):
                if os.path.isdir(os.path.join(path, folder, subfolder)):
                    if MTMC:
                        #copy the file named mtmc.txt to another txt file named the cam name for the seq 
                        os.system("cp " + os.path.join(path, folder, subfolder, "mtmc.txt") + " " + os.path.join(outputPath, cams[seq][int(subfolder.split("_")[0])] + ".txt"))

else:
    #copy the file named mtmc.txt to another txt file named the cam name for the seq 
    os.system("cp " + os.path.join(path, "mtmc.txt") + " " + os.path.join(outputPath, cams[seq][0] + ".txt"))
