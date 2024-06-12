from EasyOCR import easyocr

reader = easyocr.Reader(['en'])

image_path = "./images/test_image_1.png"
result = reader.readtext(image_path, detail=0, paragraph=True)

print(result)