import PIL.ExifTags
import PIL.Image
import numpy as np
import  cv2

#Get exif data in order to get focal length.
exif_img = PIL.Image.open("imag/Monopoly/view1.png")
exif_data = {
 PIL.ExifTags.TAGS[k]:v
 for k, v in exif_img._getexif().items()
 if k in PIL.ExifTags.TAGS}
#Get focal length in tuple form
focal_length_exif = exif_data['FocalLength']
#Get focal length in decimal form
focal_length = focal_length_exif[0]/focal_length_exif[1]
print(focal_length)
# np.save("./camera_params/FocalLength", focal_length)