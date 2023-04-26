import argparse
import os


if __name__ == '__main__':
    # Parse arguments
    parser = argparse.ArgumentParser(
        description="evaluate the tracking")
    parser.add_argument("--seq", type=str)
    parser.add_argument("--endToEnd", type=bool, default = False)
    parser.add_argument("--csv_name", type=str, default = "mtmc.csv")
    parser.add_argument("--input", type=str)
    parser.add_argument("--output", type=str)
    parser.add_argument("--evaluated", type=bool, default = False)
    parser.add_argument("--MTMC", type=bool, default = False)
    parser.add_argument("--evalOutput", type=str, default = "/ghome/group03/mcv-m6-2023-team6/week5/Results/TrackEvalResults")
    args = parser.parse_args()

   
    cams = {"S01":['c001', 'c002', 'c003', 'c004', 'c005'], "S03": ['c010', 'c011', 'c012', 'c013', 'c014', 'c015'], "S04": ['c016', 'c017', 'c018', 'c019', 'c020','c021', 'c022', 'c023', 'c024', 'c025', 'c026', 'c027', 'c028', 'c029', 'c030', 'c031', 'c032',  'c033', 'c034', 'c035','c036',  'c037', 'c038', 'c039','c040']}

    if not args.evaluated:
        print("Evaluating the tracking")
        if args.endToEnd:
            print("Using EndToEnd")
            for folder in os.listdir(args.input):
                if os.path.isdir(os.path.join(args.input, folder)):
                    #copy the file named mtmc.txt to another txt file named the cam name for the seq 
                    os.remove(os.path.join(args.output + "data/", cams[args.seq][int(folder.split("_")[0])] + ".txt"))
                    if args.MTMC:
                        os.system("cp " + os.path.join(args.input, folder, "mtmc.txt") + " " + os.path.join(args.output + "data/", cams[args.seq][int(folder.split("_")[0])] + ".txt"))
                    else:
                        os.system("cp " + os.path.join(args.input, folder, "mot.txt") + " " + os.path.join(args.output + "data/", cams[args.seq][int(folder.split("_")[0])] + ".txt"))

        else:
            print("Not EndToEnd")
            for file in os.listdir(args.input):
                if file.endswith(".txt"):
                    #copy the file named mtmc.txt to another txt file named the cam name for the seq 
                    # delete the file output txt file before, if it already exists
                    if args.MTMC:
                        os.remove(os.path.join(args.output + "data/", cams[args.seq][int(file.split("_")[0])] + ".txt"))
                        os.system("cp " + os.path.join(args.input, file) + " " + os.path.join(args.output + "data/", cams[args.seq][int(file.split("_")[0])] + ".txt"))

    else:
        print("Moving the csv")
        # copy the generated csv file to the folder 
        #get the name of the csv file in the output folder
        files = os.listdir(args.output)
        for file in files:
            if file.endswith(".csv"):
                os.system("cp " + os.path.join(args.output, file) + " " + os.path.join(args.evalOutput, args.csv_name))



    
