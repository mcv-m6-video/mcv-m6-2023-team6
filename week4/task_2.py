import argparse
from .datasets import AICityDataset

def task2(args):
    dataset = AICityDataset(args.dataset_path, args.sequences)
    pass



if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset_path', type=str, default="/ghome/group03/dataset/aic19-track1-mtmc-train",
                        help='dataset')
    parser.add_argument('--sequences', type=str, default="S03", help='sequences')

    parser.add_argument('--results_path', type=str, default='Results/Task1_2/',
                        help='path to save results in a csv file')
    parser.add_argument('--visualize', type=bool, default=True)

    args = parser.parse_args()


    task2(args)