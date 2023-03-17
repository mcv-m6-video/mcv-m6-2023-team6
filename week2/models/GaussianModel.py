import os
import cv2
import numpy as np
import concurrent.futures
import numexpr as ne

from week2.models.BaseModel import BaseModel


class GaussianModel(BaseModel):

    def __init__(self, video_path, num_frames, checkpoint=None):
        super().__init__(video_path, num_frames, checkpoint)
        # 2 modes
        self.mean = None
        self.std = None

        self.base = os.path.join(os.getcwd(), "checkpoints", "GaussianModel")

    def compute_parameters(self):
        """
            Function called after first X% of images are saved in self.images
            The values computed here will be used afterwards to compute the foreground
        """
        self.mean = np.mean(self.images, axis=-1, dtype=np.float32)
        print("Mean computed successfully.")
        self.std = np.std(self.images, axis=-1, dtype=np.float32)
        print("Standard deviation computed successfully.")

    """ NOT TESTED YET
    @staticmethod
    @njit
    def compute_mean_std(images):
        mean = np.mean(images, axis=-1)
        std = np.std(images, axis=-1)
        return mean, std

    def compute_parameters(self):
        self.mean, self.std = self.compute_mean_std(self.images)
        print("Mean and standard deviation computed successfully.")
    """
    def compute_next_foreground(self):
        """
            Function to compute the foreground. Values computed in function 'compute_parameters'
            are available to use.
        """
        if not self.modeled:
            print("[ERROR] Background has not been modeled yet.")
            return None

        success, I = self.cap.read()
        if not success:
            return None

        abs_diff = np.abs(I - self.mean)
        foreground = ne.evaluate("abs_diff * (std + 2)",
                                 local_dict={"abs_diff": abs_diff, "std": self.std})
        return foreground.astype(np.uint8) * 255, I

    def save_checkpoint(self):
        """
            Save info of the modeled background
        """
        if not os.path.exists(f"{self.base}/{self.checkpoint}"):
            os.makedirs(f"{self.base}/{self.checkpoint}")

        np.save(f"{self.base}/{self.checkpoint}/mean.npy", self.mean)
        np.save(f"{self.base}/{self.checkpoint}/std.npy", self.std)
        cv2.imwrite(f"{self.base}/{self.checkpoint}/mean.png", self.mean)
        cv2.imwrite(f"{self.base}/{self.checkpoint}/std.png", self.std)

        assert (np.load(f"{self.base}/{self.checkpoint}/mean.npy") == self.mean).all()
        assert (np.load(f"{self.base}/{self.checkpoint}/std.npy") == self.std).all()

    def load_checkpoint(self):
        """
            Load info of the modeled background
        """
        mean_path = f"{self.base}/{self.checkpoint}/mean.npy"
        std_path = f"{self.base}/{self.checkpoint}/std.npy"
        if not os.path.exists(mean_path) or not os.path.exists(std_path):
            return -1
        self.mean = np.load(mean_path)
        self.std = np.load(std_path)
        print("Checkpoint loaded.")