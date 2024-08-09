# SPUR24

# UNet
- Code: UNet.ipynb
- Images: /data/UNet_Data_Bigger (on https://py.lavlab.mcw.edu/)
	○ 512x512 images derived from 5120x5120 squares of the original histology
	○ Gathered with bigger_data_gathering.ipynb
	○ Folder names represent the the Folder ID of the patient in Omero
- Masks: /data/UNet_Data_Bigger/combined_masks_fixed
	○ 0=no cancer, 1=grade 3, 2=grade 4 FG, 3=grade 4 CG, 4=grade 5
	○ File name meaning: {folder}_{image_id}_{tile_id}_mask_{contains_g3}_{contains_g4fg}_{contains_g4cg}_{contains_g5}.png
	○ Images can be paired with masks using the file names. I do this in the fourth cell of UNet.ipynb
- Progress: The model overfits while training. The best scores achieved on the testing data were dice of 0.229, IoU of 0.2018, and accuracy of 0.3458. A report of the best run can be found here: https://api.wandb.ai/links/sarah-i-rubenstein-Medical College of Wisconsin/n719j5v7. The saved model from this run can be found here: https://drive.google.com/file/d/135BQsa6c1XsJqs7XbmQzXZsfy4AQYdP5/view?usp=sharing (file too big for GitHub lol)


# EfficientNet
- I managed to get Luke's code (his code and paper are located here: https://github.com/yoderj/ur-luke-josiah) running, but I could not replicate his results. I was having a problem with the model values becoming NaN. My slight adaptations of his code (for my environment) are in efnet.ipynb in my GitHub
I also played around with transfer learning using a model pretrained on ImageNet, but it didn't look promising. My attempt is in efficientnet_TL.ipynb
