import csv
from torch.utils.data import Dataset
from PIL import Image
Image.MAX_IMAGE_PIXELS = None  # Disable the limit
import torchvision.transforms as transforms

class ImageRegistrationDataset(Dataset):
    def __init__(self, csv_file, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.csv_file = csv_file
        self.hist_label_paths = []
        self.microus_label_paths = []
        self._load_data()
        # Define transformations
        self.image_transform = transforms.Compose([
          transforms.Resize((512, 512)),  # Resize to (512, 512)
          transforms.ToTensor(),  # Convert to PyTorch tensor
          # transforms.Normalize([0.5, 0.5, 0.5], [0.5,0.5,0.5]),  # Normalize by dividing with 255 (assuming three-channel image)
        ])
        self.mask_transform = transforms.Compose([
          transforms.Resize((512, 512)),  # Resize to (512, 512)
          transforms.ToTensor(),  # Convert to PyTorch tensor
          # transforms.Normalize([0.5, 0.5, 0.5], [0.5,0.5,0.5]),  # Normalize by dividing with 255 (assuming three-channel image)
        ])

    def _load_data(self):
        with open(self.csv_file, "r") as csvfile:
            csv_reader = csv.reader(csvfile)
            next(csv_reader)  # Skip the header row
            for row in csv_reader:
                hist_path, micro_path = row
                self.hist_label_paths.append(hist_path)
                self.microus_label_paths.append(micro_path)
                

    def __len__(self):
        return len(self.microus_label_paths)

    def __getitem__(self, idx):
        hist_mask_name = self.hist_label_paths[idx]
        hist_image_name = hist_mask_name.replace("label", "slice")
        micro_mask_name = self.hist_label_paths[idx].replace("hist", "microus")
        micro_image_name = micro_mask_name.replace("label", "slice")
        
        
        hist_mask = Image.open(hist_mask_name).convert('RGB')
        hist_image = Image.open(hist_image_name).convert('RGB')
        micro_mask = Image.open(micro_mask_name).convert('RGB')
        micro_image = Image.open(micro_image_name).convert('RGB')
        
        if self.mask_transform:
            hist_mask = self.mask_transform(hist_mask)
            micro_mask = self.mask_transform(micro_mask)
            
        if self.image_transform:
            hist_image = self.image_transform(hist_image)
            micro_image = self.mask_transform(micro_image)

        # Convert mask to 0 or 1 (assuming background is 0 and object is non-zero)
        hist_mask = (hist_mask > 0).float()  # Convert to float tensor with 0 or 1 values
        micro_mask = (micro_mask > 0).float()
        return micro_image, micro_mask, hist_image, hist_mask
    
