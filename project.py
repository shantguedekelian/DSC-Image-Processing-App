"""
DSC 20 Project
Name(s): Jaden Vo & Shant Geudekelian
PID(s):  A17897553 & A17975990
Sources: None
"""

import numpy as np
import os
from PIL import Image

NUM_CHANNELS = 3


# --------------------------------------------------------------------------- #

def img_read_helper(path):
    """
    Creates an RGBImage object from the given image file
    """
    # Open the image in RGB
    img = Image.open(path).convert("RGB")
    # Convert to numpy array and then to a list
    matrix = np.array(img).tolist()
    # Use student's code to create an RGBImage object
    return RGBImage(matrix)


def img_save_helper(path, image):
    """
    Saves the given RGBImage instance to the given path
    """
    # Convert list to numpy array
    img_array = np.array(image.get_pixels())
    # Convert numpy array to PIL Image object
    img = Image.fromarray(img_array.astype(np.uint8))
    # Save the image object to path
    img.save(path)


# --------------------------------------------------------------------------- #

# Part 1: RGB Image #
class RGBImage:
    """
    Represents an image in RGB format
    """

    def __init__(self, pixels):
        """
        Initializes a new RGBImage object

        # Test with non-rectangular list
        >>> pixels = [
        ...              [[255, 255, 255], [255, 255, 255]],
        ...              [[255, 255, 255]]
        ...          ]
        >>> RGBImage(pixels)
        Traceback (most recent call last):
        ...
        TypeError

        # Test instance variables
        >>> pixels = [
        ...              [[255, 255, 255], [0, 0, 0]]
        ...          ]
        >>> img = RGBImage(pixels)
        >>> img.pixels
        [[[255, 255, 255], [0, 0, 0]]]
        >>> img.num_rows
        1
        >>> img.num_cols
        2
        """
        # YOUR CODE GOES HERE #
        # Raise exceptions here
        max_len = 3
        if type(pixels) != list:
            raise TypeError()
        elif len(pixels) < 1:
            raise TypeError()
        for lst in pixels:
            if type(lst) != list:
                raise TypeError()
            elif len(lst) < 1:
                raise TypeError()
        if not all([True if len(itm) == len(pixels[0]) else False \
            for itm in pixels]):
            raise TypeError()
        if not all([True if len(nlst) == len(lst[0]) else False \
            for lst in pixels for nlst in lst]):
            raise TypeError()
        if not all([True if len(nlst) == max_len else False \
            for lst in pixels for nlst in lst]):
            raise TypeError()


        self.pixels = pixels
        self.num_rows = len(pixels)
        self.num_cols = len(pixels[0])

    def size(self):
        """
        Returns the size of the image in (rows, cols) format

        # Make sure to complete __init__ first
        >>> pixels = [
        ...              [[255, 255, 255], [0, 0, 0]]
        ...          ]
        >>> img = RGBImage(pixels)
        >>> img.size()
        (1, 2)
        """
        # YOUR CODE GOES HERE #
        return (self.num_rows, self.num_cols)

    def get_pixels(self):
        """
        get_pixels(pixels[1:])
        Returns a copy of the image pixel array

        # Make sure to complete __init__ first
        >>> pixels = [
        ...              [[255, 255, 255], [0, 0, 0]]
        ...          ]
        >>> img = RGBImage(pixels)
        >>> img_pixels = img.get_pixels()

        # Check if this is a deep copy
        >>> img_pixels                               # Check the values
        [[[255, 255, 255], [0, 0, 0]]]
        >>> id(pixels) != id(img_pixels)             # Check outer list
        True
        >>> id(pixels[0]) != id(img_pixels[0])       # Check row
        True
        >>> id(pixels[0][0]) != id(img_pixels[0][0]) # Check pixel
        True

        >>> pixels = [
        ...              [[255, 255, 255], [0, 0, 0]],
        ...              [[100, 200, 212], [1, 1, 1]]
        ...          ]
        >>> img = RGBImage(pixels)
        >>> img.get_pixels()
        [[[255, 255, 255], [0, 0, 0]], [[100, 200, 212], [1, 1, 1]]]
        """
        # YOUR CODE GOES HERE #
        return [[nlst[:] for nlst in lst] for lst in self.pixels]



    def copy(self):
        """
        Returns a copy of this RGBImage object

        # Make sure to complete __init__ first
        >>> pixels = [
        ...              [[255, 255, 255], [0, 0, 0]]
        ...          ]
        >>> img = RGBImage(pixels)
        >>> img_copy = img.copy()

        # Check that this is a new instance
        >>> id(img_copy) != id(img)
        True
        """
        # YOUR CODE GOES HERE #
        return RGBImage(self.get_pixels())

    def get_pixel(self, row, col):
        """
        Returns the (R, G, B) value at the given position

        # Make sure to complete __init__ first
        >>> pixels = [
        ...              [[255, 255, 255], [0, 0, 0]]
        ...          ]
        >>> img = RGBImage(pixels)

        # Test with an invalid index
        >>> img.get_pixel(1, 0)
        Traceback (most recent call last):
        ...
        ValueError

        # Run and check the returned value
        >>> img.get_pixel(0, 0)
        (255, 255, 255)
        """
        # YOUR CODE GOES HERE #
        if type(row) != int:
            raise TypeError()
        elif row < 0:
            raise ValueError()
        if type(col) != int:
            raise TypeError()
        elif col < 0:
            raise ValueError()
        if len(self.pixels) <= row:
            raise ValueError()
        elif len(self.pixels[row]) <= col:
            raise ValueError()

        return tuple(self.pixels[row][col])


    def set_pixel(self, row, col, new_color):
        """
        Sets the (R, G, B) value at the given position

        # Make sure to complete __init__ first
        >>> pixels = [
        ...              [[255, 255, 255], [0, 0, 0]]
        ...          ]
        >>> img = RGBImage(pixels)

        # Test with an invalid new_color tuple
        >>> img.set_pixel(0, 0, (256, 0, 0))
        Traceback (most recent call last):
        ...
        ValueError

        # Check that the R/G/B value with negative is unchanged
        >>> img.set_pixel(0, 0, (-1, 0, 0))
        >>> img.pixels
        [[[255, 0, 0], [0, 0, 0]]]
        """
        # YOUR CODE GOES HERE #
        max_hex = 255
        if type(row) != int:
            raise TypeError()
        elif row < 0:
            raise ValueError()
        if type(col) != int:
            raise TypeError()
        elif col < 0:
            raise ValueError()
        if len(self.pixels) <= row:
            raise ValueError()
        elif len(self.pixels[row]) <= col:
            raise ValueError()
        for value in new_color:
            if value > max_hex:
                raise ValueError()

        self.pixels[row][col] = list(map(lambda color: color, \
            list(map(lambda idx: self.pixels[row][col][idx] \
            if new_color[idx] < 0 else \
            new_color[idx],range(len(new_color))))))

# Part 2: Image Processing Template Methods #
class ImageProcessingTemplate:
    """
    Contains assorted image processing methods
    Intended to be used as a parent class
    """

    def __init__(self):
        """
        Creates a new ImageProcessingTemplate object

        # Check that the cost was assigned
        >>> img_proc = ImageProcessingTemplate()
        >>> img_proc.cost
        0
        """
        # YOUR CODE GOES HERE #
        self.cost = 0

    def get_cost(self):
        """
        Returns the current total incurred cost

        # Check that the cost value is returned
        >>> img_proc = ImageProcessingTemplate()
        >>> img_proc.cost = 50 # Manually modify cost
        >>> img_proc.get_cost()
        50
        """
        # YOUR CODE GOES HERE #
        return self.cost

    def negate(self, image):
        """
        Returns a negated copy of the given image

        # Check if this is returning a new RGBImage instance
        >>> img_proc = ImageProcessingTemplate()
        >>> pixels = [
        ...              [[255, 255, 255], [0, 0, 0]]
        ...          ]
        >>> img = RGBImage(pixels)
        >>> img_negate = img_proc.negate(img)
        >>> id(img) != id(img_negate) # Check for new RGBImage instance
        True

        # The following is a description of how this test works
        # 1 Create a processor
        # 2/3 Read in the input and expected output
        # 4 Modify the input
        # 5 Compare the modified and expected
        # 6 Write the output to file
        # You can view the output in the img/out/ directory
        >>> img_proc = ImageProcessingTemplate()                            # 1
        >>> img = img_read_helper('img/test_image_32x32.png')                 # 2
        >>> img_exp = img_read_helper('img/exp/test_image_32x32_negate.png')  # 3
        >>> img_negate = img_proc.negate(img)                               # 4
        >>> img_negate.pixels == img_exp.pixels # Check negate output       # 5
        True
        >>> img_save_helper('img/out/test_image_32x32_negate.png', img_negate)# 6
        """
        # YOUR CODE GOES HERE #

        return RGBImage([[[255-val for val in col] for col in row] \
            for row in image.get_pixels()])


    def grayscale(self, image):
        """
        Returns a grayscale copy of the given image

        # See negate for info on this test
        # You can view the output in the img/out/ directory
        >>> img_proc = ImageProcessingTemplate()
        >>> img = img_read_helper('img/test_image_32x32.png')
        >>> img_exp = img_read_helper('img/exp/test_image_32x32_gray.png')
        >>> img_gray = img_proc.grayscale(img)
        >>> img_gray.pixels == img_exp.pixels # Check grayscale output
        True
        >>> img_save_helper('img/out/test_image_32x32_gray.png', img_gray)
        """
        # YOUR CODE GOES HERE #

        return RGBImage([[[sum(col)//3 for val in col] for col in row] \
            for row in image.get_pixels()])

    def rotate_180(self, image):
        """
        Returns a rotated version of the given image

        # See negate for info on this test
        # You can view the output in the img/out/ directory
        >>> img_proc = ImageProcessingTemplate()
        >>> img = img_read_helper('img/test_image_32x32.png')
        >>> img_exp = img_read_helper('img/exp/test_image_32x32_rotate.png')
        >>> img_rotate = img_proc.rotate_180(img)
        >>> img_rotate.pixels == img_exp.pixels # Check rotate_180 output
        True
        >>> img_save_helper('img/out/test_image_32x32_rotate.png', img_rotate)
        """
        # YOUR CODE GOES HERE #

        return RGBImage([[col for col in row[::-1]] \
            for row in image.get_pixels()[::-1]])

    def get_average_brightness(self, image):
        """
        Returns the average brightness for the given image

        >>> img_proc = ImageProcessingTemplate()
        >>> img = img_read_helper('img/test_image_32x32.png')
        >>> img_proc.get_average_brightness(img)
        86
        """
        # YOUR CODE GOES HERE #
        pix_avg = [sum(col)//3 for row in image.get_pixels() for col in row for val in col]
        return sum(pix_avg)//len([val for row in image.get_pixels() for col in row for val in col])


    def adjust_brightness(self, image, intensity):
        """
        Returns a new image with adjusted brightness level

        >>> img_proc = ImageProcessingTemplate()
        >>> img = img_read_helper('img/test_image_32x32.png')
        >>> img_exp = img_read_helper('img/exp/test_image_32x32_adjusted.png')
        >>> img_adjust = img_proc.adjust_brightness(img, 75)
        >>> img_adjust.pixels == img_exp.pixels # Check adjust_brightness
        True
        >>> img_save_helper('img/out/test_image_32x32_adjusted.png', img_adjust)
        """
        # YOUR CODE GOES HERE #

        return RGBImage([[[255 if val+intensity > 255 else \
            0 if val+intensity < 0 else val+intensity for val in col] \
            for col in row] for row in image.get_pixels()])


    def blur(self, image):
        """
        Returns a new image with the pixels blurred

        >>> img_proc = ImageProcessingTemplate()
        >>> img = img_read_helper('img/test_image_32x32.png')
        >>> img_exp = img_read_helper('img/exp/test_image_32x32_blur.png')
        >>> img_blur = img_proc.blur(img)
        >>> img_blur.pixels == img_exp.pixels # Check blur
        True
        >>> img_save_helper('img/out/test_image_32x32_blur.png', img_blur)
        """
        # YOUR CODE GOES HERE #
        copy = image.copy()

        for row in range(copy.size()[0]):
            for col in range(copy.size()[1]):
                rsum, gsum, bsum = 0, 0, 0
                pixels = 0
                if col-1>=0: #item to the left
                    rsum += image.get_pixel(row, col-1)[0]
                    gsum += image.get_pixel(row, col-1)[1]
                    bsum += image.get_pixel(row, col-1)[-1]
                    pixels += 1
                if col+1<copy.size()[1]: #item to the right
                    rsum += image.get_pixel(row, col+1)[0]
                    gsum += image.get_pixel(row, col+1)[1]
                    bsum += image.get_pixel(row, col+1)[-1]
                    pixels += 1
                if row-1>=0: #row above
                    if col-1>=0: #item to the left
                        rsum += image.get_pixel(row-1, col-1)[0]
                        gsum += image.get_pixel(row-1, col-1)[1]
                        bsum += image.get_pixel(row-1, col-1)[-1]
                        pixels += 1
                    if col+1<copy.size()[1]: #item to the right
                        rsum += image.get_pixel(row-1, col+1)[0]
                        gsum += image.get_pixel(row-1, col+1)[1]
                        bsum += image.get_pixel(row-1, col+1)[-1]
                        pixels += 1
                    #item directly above    
                    rsum += image.get_pixel(row-1, col)[0]
                    gsum += image.get_pixel(row-1, col)[1]
                    bsum += image.get_pixel(row-1, col)[-1]
                    pixels += 1
                if row+1<copy.size()[0]: #row below    
                    if col-1>=0: #item to the left
                        rsum += image.get_pixel(row+1, col-1)[0]
                        gsum += image.get_pixel(row+1, col-1)[1]
                        bsum += image.get_pixel(row+1, col-1)[-1]
                        pixels += 1
                    if col+1<copy.size()[1]: #item to the right
                        rsum += image.get_pixel(row+1, col+1)[0]
                        gsum += image.get_pixel(row+1, col+1)[1]
                        bsum += image.get_pixel(row+1, col+1)[-1]
                        pixels += 1
                    #item directly below    
                    rsum += image.get_pixel(row+1, col)[0]
                    gsum += image.get_pixel(row+1, col)[1]
                    bsum += image.get_pixel(row+1, col)[-1]
                    pixels += 1

                #add itself
                rsum += image.get_pixel(row, col)[0]
                gsum += image.get_pixel(row, col)[1]
                bsum += image.get_pixel(row, col)[-1]
                pixels += 1

                rsum, gsum, bsum = rsum//pixels, gsum//pixels, bsum//pixels
                copy.set_pixel(row, col, (rsum, gsum, bsum)) 

        return copy.copy()





# Part 3: Standard Image Processing Methods #
class StandardImageProcessing(ImageProcessingTemplate):
    """
    Represents a standard tier of an image processor
    """

    def __init__(self):
        """
        Creates a new StandardImageProcessing object

        # Check that the cost was assigned
        >>> img_proc = ImageProcessingTemplate()
        >>> img_proc.cost
        0
        """
        # YOUR CODE GOES HERE #
        self.cost = 0
        self.coupons = 0

    def negate(self, image):
        """
        Returns a negated copy of the given image

        # Check the expected cost
        >>> img_proc = StandardImageProcessing()
        >>> img_in = img_read_helper('img/square_32x32.png')
        >>> negated = img_proc.negate(img_in)
        >>> img_proc.get_cost()
        5

        # Check that negate works the same as in the parent class
        >>> img_proc = StandardImageProcessing()
        >>> img = img_read_helper('img/test_image_32x32.png')
        >>> img_exp = img_read_helper('img/exp/test_image_32x32_negate.png')
        >>> img_negate = img_proc.negate(img)
        >>> img_negate.pixels == img_exp.pixels # Check negate output
        True
        """
        # YOUR CODE GOES HERE #
        if self.coupons > 0:
            self.cost += 0
        else:
            self.cost += 5
        return super().negate(image)

    def grayscale(self, image):
        """
        Returns a grayscale copy of the given image

        """
        # YOUR CODE GOES HERE #
        if self.coupons > 0:
            self.cost += 0
        else:
            self.cost += 6
        return super().greyscale(image)

    def rotate_180(self, image):
        """
        Returns a rotated version of the given image
        """
        # YOUR CODE GOES HERE #
        if self.coupons > 0:
            self.cost += 0
        else:
            self.cost += 5
        return super().rotate_180(image)

    def adjust_brightness(self, image, intensity):
        """
        Returns a new image with adjusted brightness level
        """
        # YOUR CODE GOES HERE #
        if self.coupons > 0:
            self.cost += 0
        else:
            self.cost += 1
        return super().adjust_brightness(image, intensity)

    def blur(self, image):
        """
        Returns a new image with the pixels blurred
        """
        # YOUR CODE GOES HERE #
        if self.coupons > 0:
            self.cost += 0
        else:
            self.cost += 5
        return super().blur(image)

    def redeem_coupon(self, amount):
        """
        Makes the given number of methods calls free

        # Check that the cost does not change for a call to negate
        # when a coupon is redeemed
        >>> img_proc = StandardImageProcessing()
        >>> img = img_read_helper('img/test_image_32x32.png')
        >>> img_proc.redeem_coupon(1)
        >>> img = img_proc.rotate_180(img)
        >>> img_proc.get_cost()
        0
        """
        # YOUR CODE GOES HERE #
        if type(amount) != int:
            raise TypeError()
        elif amount <= 0:
            raise ValueError()

        self.coupons += amount




# Part 4: Premium Image Processing Methods #
class PremiumImageProcessing(ImageProcessingTemplate):
    """
    Represents a paid tier of an image processor
    """

    def __init__(self):
        """
        Creates a new PremiumImageProcessing object

        # Check the expected cost
        >>> img_proc = PremiumImageProcessing()
        >>> img_proc.get_cost()
        50
        """
        # YOUR CODE GOES HERE #
        self.cost = 50

    def chroma_key(self, chroma_image, background_image, color):
        """
        Returns a copy of the chroma image where all pixels with the given
        color are replaced with the background image.

        # Check output
        >>> img_proc = PremiumImageProcessing()
        >>> img_in = img_read_helper('img/square_32x32.png')
        >>> img_in_back = img_read_helper('img/test_image_32x32.png')
        >>> color = (255, 255, 255)
        >>> img_exp = img_read_helper('img/exp/square_32x32_chroma.png')
        >>> img_chroma = img_proc.chroma_key(img_in, img_in_back, color)
        >>> img_chroma.pixels == img_exp.pixels # Check chroma_key output
        True
        >>> img_save_helper('img/out/square_32x32_chroma.png', img_chroma)
        """
        # YOUR CODE GOES HERE #
        if type(chroma_image) != RGBImage:
            return TypeError()
        if type(background_image) != RGBImage:
            return TypeError()
        if chroma_image.size() != background_image.size():
            return ValueError()

        return RGBImage([[background_image.get_pixels()[row][col] \
            if chroma_image.get_pixel(row, col) == color \
            else chroma_image.get_pixels()[row][col] 
            for col in range(chroma_image.size()[1])] \
            for row in range(chroma_image.size()[0])])


    def sticker(self, sticker_image, background_image, x_pos, y_pos):
        """
        Returns a copy of the background image where the sticker image is
        placed at the given x and y position.

        # Test with out-of-bounds image and position size
        >>> img_proc = PremiumImageProcessing()
        >>> img_sticker = img_read_helper('img/square_6x6.png')
        >>> img_back = img_read_helper('img/test_image_32x32.png')
        >>> x, y = (31, 0)
        >>> img_proc.sticker(img_sticker, img_back, x, y)
        Traceback (most recent call last):
        ...
        ValueError

        # Check output
        >>> img_proc = PremiumImageProcessing()
        >>> img_sticker = img_read_helper('img/square_6x6.png')
        >>> img_back = img_read_helper('img/test_image_32x32.png')
        >>> x, y = (3, 3)
        >>> img_exp = img_read_helper('img/exp/test_image_32x32_sticker.png')
        >>> img_combined = img_proc.sticker(img_sticker, img_back, x, y)
        >>> img_combined.pixels == img_exp.pixels # Check sticker output
        True
        >>> img_save_helper('img/out/test_image_32x32_sticker.png', img_combined)
        """
        # YOUR CODE GOES HERE #
        if type(sticker_image) != RGBImage:
            raise TypeError()
        if type(background_image) != RGBImage:
            raise TypeError()
        if sticker_image.size()[0] > background_image.size()[0] or \
            sticker_image.size()[1] > background_image.size()[1]:
            raise ValueError()
        if type(x_pos) != int or type(y_pos) != int:
            raise TypeError()
        if sticker_image.size()[0] + x_pos > background_image.size()[0] or \
            sticker_image.size()[1] + y_pos > background_image.size()[1]:
            raise ValueError()

        copy = background_image.copy()

        for row in range(len(sticker_image.get_pixels())):
            for col in range(len(sticker_image.get_pixels()[row])):
                copy.pixels[row+y_pos][col+x_pos] = \
                    sticker_image.pixels[row][col] 
        return copy

    def edge_highlight(self, image):
        """
        Returns a new image with the edges highlighted

        # Check output
        >>> img_proc = PremiumImageProcessing()
        >>> img = img_read_helper('img/test_image_32x32.png')
        >>> img_edge = img_proc.edge_highlight(img)
        >>> img_exp = img_read_helper('img/exp/test_image_32x32_edge.png')
        >>> img_exp.pixels == img_edge.pixels # Check edge_highlight output
        True
        >>> img_save_helper('img/out/test_image_32x32_edge.png', img_edge)
        """
        # YOUR CODE GOES HERE #

        highlighted = [[sum(pixel)//3 for pixel in row] for row in image.get_pixels()]
        kernel = [[-1, -1, -1],[-1,  8, -1],[-1, -1, -1]]
        copy = [[0] * len(row) for row in highlighted]

        for row in range(len(highlighted)):
            for col in range(len(highlighted[0])): #[row][col] gets every pixel once
                total = 0

                for pixel_row in range(row-1, row+2): #row before, self, and below
                    for pixel_col in range(col-1, col+2): #col before, self, and below

                        if 0 <= pixel_row and pixel_row < len(highlighted): #is a valid row
                            if 0 <= pixel_col and pixel_col < len(highlighted[0]): #is a valid col
                                        
                                if pixel_row == row and pixel_col == col: #apply mask value
                                    total += highlighted[pixel_row][pixel_col] * 8
                                else:
                                    total -= highlighted[pixel_row][pixel_col]


                copy[row][col] = max(0, min(total, 255))        

        return RGBImage([[[col]*3 for col in row] for row in copy])

    

# Part 5: Image KNN Classifier #
class ImageKNNClassifier:
    """
    Represents a simple KNNClassifier
    """

    def __init__(self, k_neighbors):
        """
        Creates a new KNN classifier object
        """
        # YOUR CODE GOES HERE #
        self.k_neighbors = k_neighbors
        self.data = []

    def fit(self, data):
        """
        Stores the given set of data and labels for later
        """
        # YOUR CODE GOES HERE #
        if len(data) < self.k_neighbors:
            raise ValueError()
        self.data = [(image, label) for image,label in data]


    def distance(self, image1, image2):
        """
        Returns the distance between the given images

        >>> img1 = img_read_helper('img/steve.png')
        >>> img2 = img_read_helper('img/knn_test_img.png')
        >>> knn = ImageKNNClassifier(3)
        >>> knn.distance(img1, img2)
        15946.312896716909
        """
        # YOUR CODE GOES HERE #
        if type(image1) != RGBImage or type(image2) != RGBImage:
            raise TypeError()
        elif image1.size() != image2.size():
            raise ValueError()

        return sum([(image1.get_pixel(row,col)[index]- \
            image2.get_pixel(row,col)[index])**2 \
            for row in range(len(image1.get_pixels())) \
            for col in range(len(image1.get_pixels()[row])) \
            for index in range(3)]) ** 0.5


    def vote(self, candidates):
        """
        Returns the most frequent label in the given list

        >>> knn = ImageKNNClassifier(3)
        >>> knn.vote(['label1', 'label2', 'label2', 'label2', 'label1'])
        'label2'
        """
        # YOUR CODE GOES HERE #
        counts = {}
        for item in candidates:
            if item in counts:
                counts[item] += 1
            else:
                counts[item] = 1
        return max(counts, key=counts.get)


    def predict(self, image):
        """
        Predicts the label of the given image using the labels of
        the K closest neighbors to this image

        The test for this method is located in the knn_tests method below
        """
        # YOUR CODE GOES HERE #
        if not self.data:
            raise ValueError()

        dists = {self.distance(image, img): label for img,label in self.data}

        sorted_dists = dict(sorted(dists.items()))
        return self.vote(list(sorted_dists.values())[:self.k_neighbors])



def knn_tests(test_img_path):
    """
    Function to run knn tests

    >>> knn_tests('img/knn_test_img.png')
    'nighttime'
    """
    # Read all of the sub-folder names in the knn_data folder
    # These will be treated as labels
    path = 'knn_data'
    data = []
    for label in os.listdir(path):
        label_path = os.path.join(path, label)
        # Ignore non-folder items
        if not os.path.isdir(label_path):
            continue
        # Read in each image in the sub-folder
        for img_file in os.listdir(label_path):
            train_img_path = os.path.join(label_path, img_file)
            img = img_read_helper(train_img_path)
            # Add the image object and the label to the dataset
            data.append((img, label))

    # Create a KNN-classifier using the dataset
    knn = ImageKNNClassifier(5)

    # Train the classifier by providing the dataset
    knn.fit(data)

    # Create an RGBImage object of the tested image
    test_img = img_read_helper(test_img_path)

    # Return the KNN's prediction
    predicted_label = knn.predict(test_img)
    return predicted_label
