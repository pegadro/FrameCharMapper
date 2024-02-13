import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont


def map_value_to_char(value, scale):
    factor = len(scale) / 256.0
    char_index = int(value * factor)
    
    return scale[char_index]


def get_transformed_frame_size(resize):
    ret, frame = cap.read()
    
    if not ret:
        print("Unable to capture video")    
    
    frame = cv2.flip(frame, 1)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame_pil = Image.fromarray(frame_rgb)
    frame_pil = frame_pil.convert('L').resize((frame_pil.size[0] // resize, frame_pil.size[1] // resize))
    frame_array = np.array(frame_pil)
    
    return frame_array.shape
    

if __name__ == "__main__":
    chars_scale = " .:-=+*#%@"    
    cap = cv2.VideoCapture(0)
    map_value_to_char_vectorized = np.vectorize(map_value_to_char)
    resize = 15
    
    width_per_char, height_per_char = 10, 10
    transformed_frame_size = get_transformed_frame_size(resize)
    width, height = width_per_char * transformed_frame_size[1], height_per_char * transformed_frame_size[0]
    
    image_char = Image.new('RGB', (width, height), 'black')
    draw = ImageDraw.Draw(image_char)
    
    transformed_frame = np.full(transformed_frame_size, "a", dtype=str)
    
    try:
        font = ImageFont.truetype("arial.ttf", 15)
    except IOError:
        font = ImageFont.load_default()
    
    while True:
        ret, frame = cap.read()
        
        if not ret:
            print("Unable to capture video")
            break
        
        frame = cv2.flip(frame, 1)
        
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_pil = Image.fromarray(frame_rgb)    
        frame_pil = frame_pil.convert('L').resize((frame_pil.size[0] // resize, frame_pil.size[1] // resize))
        
        frame_array = np.array(frame_pil)
        temp_transformed_frame = map_value_to_char_vectorized(frame_array, chars_scale)
        
        indexes = np.where(transformed_frame != temp_transformed_frame)
        
        for index in zip(*indexes):
            pos_x = index[1] * width_per_char
            pos_y = index[0] * height_per_char
            draw.rectangle((pos_x, pos_y, pos_x + width_per_char, pos_y + height_per_char), fill='black')
            draw.text((pos_x, pos_y), temp_transformed_frame[index[0],index[1]], fill='white', font=font)
        
        cv2.imshow('Frame', np.array(image_char))
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        
        transformed_frame = temp_transformed_frame
        
    cap.release()
    cv2.destroyAllWindows()