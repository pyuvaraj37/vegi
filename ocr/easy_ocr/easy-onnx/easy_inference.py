from EasyOCR import easyocr
import torch
from timeit import default_timer as timer

# Initialize the EasyOCR reader
reader = easyocr.Reader(['en'])  # Initialize with the desired language

# Open a file to save the results
with open('NPU_results.txt', 'w') as file:
    for i in range(1,8):
        start = timer()
        image_path = "./images/test_image_"+str(i)+".png"
        result = reader.readtext(image_path, detail = 0, paragraph=True)
        
     # Output the OCR results
        if result:
            print(result)
            # Write the result to a file
            file.write('\n'.join(result) + '\n')
            
        # Uncomment below to also log processing time to the file
        processing_time = f"NPU-ONNX Processing Time: {timer() - start}\n"
        file.write(processing_time)



