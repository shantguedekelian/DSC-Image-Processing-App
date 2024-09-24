# DSC-Image-Processing-App
Final project for DSC20 at UCSD. Making an app with features such as: blurring, grayscale, rotating, changing brightness of any given image.

Some features of the app:

**negate (self, image)** 

![Image](https://github.com/user-attachments/assets/c648044a-d80e-4943-914a-239a01a69ed7)

The negate function returns the negative of an image (inverts pixels).

**grayscale (self, image)**

![Image](https://github.com/user-attachments/assets/b00abd62-791e-4b8f-ba10-d42a8223a249)

The grayscale function removes color from the image.

**rotate_180 (self, image)**
![Image](https://github.com/user-attachments/assets/b999e506-f5c3-4ee4-9e20-7beb0074b3cd)
The rotate_180 function rotates an image 180 degrees and returns a new image.

**adjust_brightness (self, image, intensity)**
![Image](https://github.com/user-attachments/assets/a42ec290-ff73-4e47-9be6-3c044cc0eb60)
This function allows you to change the overall brightness of the image.

**blur (self, image)**
![Image](https://github.com/user-attachments/assets/bec79256-31c9-4cd6-9bec-2a65079a8c32)
Using blur, you can blur an image by making the RGB values of each pixel the average of itself and its 8 neighbors. 

**chroma_key (self, chroma_image, background_image, color)**
![Image](https://github.com/user-attachments/assets/7bb4e6ab-9100-491f-8cdc-2d9dd48ede39)
Changes the specified pixel color to the corresponding pixel in the background image thats been chosen.

**sticker (self, sticker_image, background_image, x_pos, y_pos)**
![Image](https://github.com/user-attachments/assets/b37ba860-b8de-4658-a30e-2b21a91714e7)
Allows you to put a sticker on the background image wherever you want (top left of sticker is placed at x and y position given).

**edge_highlight (self, image)**
![Image](https://github.com/user-attachments/assets/02106428-6147-4b3d-ad73-991a9fd0b59a)
Highlights the edges of the image and makes everything else black.
