import cv2

def crop_with_gap(image, x, y, w, h, gap):
    print(f"crop x:{x} y:{y} w:{w} h:{h} gap:{gap}")
    # Calculate the new bounding box with the gap
    x_start = max(0, int(x - gap))
    y_start = max(0, int(y - gap))
    x_end = min(image.shape[1], int(x + w + gap))  # image.shape[1] is width
    y_end = min(image.shape[0], int(y + h + gap))  # image.shape[0] is height
    
    # Crop the image with the updated coordinates
    cropped_image = image[y_start:y_end, x_start:x_end]
    
    return cropped_image