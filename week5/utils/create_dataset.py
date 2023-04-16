from util import *
import os

def create_dataset(path):

    if not os.path.exists(path+'/labels'):
        os.makedirs(path+'/labels')

    frames = len(os.listdir(path+'/frames/'))

    gt = load_from_txt(path+'/gt/gt.txt')

    for f in range(1,frames+1):
        if f in gt:
            print(f)
            file = open(path+'/labels/'+str(f)+'.txt','w')
            for det in gt[f]:
                print(det)
                line = f'{det[0]} {det[1]} {det[2]} {det[3]} {det[4]} \n'
                file.write(line)

            file.close()
        else:
            with open(path+'/labels/'+str(f)+'.txt','w') as file:
                pass

create_dataset('C:/Users/AnaHarris/Documents/MASTER/M6/project/dataset/seqs/S01/c001')
