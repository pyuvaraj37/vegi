import sys
import os
import torch
from PIL import ImageGrab
from timeit import default_timer as timer
from EasyOCR import easyocr

def screenGrab(rect):
    """ Given a rectangle, return a PIL Image of that part of the screen.
        Uses ImageGrab on Windows. """
    x, y, width, height = rect
    image = ImageGrab.grab(bbox=(x, y, x + width, y + height))
    return image

reader = easyocr.Reader(['en'])  # Initialize with the desired language

if __name__ == "__main__":
    EXE = sys.argv[0]
    del(sys.argv[0])

    if len(sys.argv) != 4 or sys.argv[0] in ('--help', '-h', '-?', '/?'):
        sys.stderr.write(EXE + ": monitors section of screen for text\n")
        sys.stderr.write(EXE + ": Give x, y, width, height as arguments\n")
        sys.exit(1)

    x, y, width, height = map(int, sys.argv[:4])
    screen_rect = [x, y, width, height]
    print(EXE + ": watching " + str(screen_rect))

    # Open a file to save the results
    with open('ocr_results.txt', 'w') as file:
        while True:
            start = timer()
            image = screenGrab(screen_rect)  # Grab the area of the screen
            image_path = "./temp_screengrab.png"
            image.save(image_path)   # Save the image to a temporary file
            
            result = reader.readtext(image_path, detail=0, paragraph=True)  # OCR the image
            
            # Output the OCR results
            if result:
                print(result)
                # Write the result to a file
                file.write('\n'.join(result) + '\n')
            
            # Uncomment below to also log processing time to the file
            # processing_time = f"Processing Time: {timer() - start}\n"
            # file.write(processing_time)
