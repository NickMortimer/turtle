from PIL import Image
import os
import pandas as pd
import glob

# Function to create a large image from small images
def create_large_image(small_images,  rows, columns,tiles):

    # Calculate dimensions of the large image
    small_image = Image.open(small_images[0])
    small_width, small_height = small_image.size
    large_width = small_width * columns
    large_height = small_height * rows

    # Create a new blank large image
    large_image = Image.new("RGB", (large_width, large_height))

    # Paste small images onto the large image
    index = 0
    for t in range(tiles):
        for i in range(rows):
            for j in range(columns):
                if index < len(small_images):
                    small_image = Image.open(small_images[index])
                    large_image.paste(small_image, (j * small_width, i * small_height))
                    index=index+1

        # Save the large image
        large_image.save(f"/media/mor582/turtles/Ningaloo/surveys/train/poster/large_image{t:03}.jpg")

# Example usage
images = glob.glob("/media/mor582/turtles/Ningaloo/surveys/train/poster/images/*.jpg")
df = pd.DataFrame(images,columns=['FilePath']).sample(5000)
rows = 3  # Number of rows of small images
columns = 4  # Number of columns of small images6
create_large_image(df.FilePath.to_list(), 21,15,100 )