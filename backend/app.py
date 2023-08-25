from flask import Flask, request, jsonify
import pickle
import pandas as pd
import requests
from flask_cors import CORS, cross_origin
import cv2
import numpy as np
from datetime import datetime
import math
from PIL import Image
from werkzeug.utils import secure_filename
import os

model = pickle.load(open("my_model.pkl", "rb"))

UPLOAD_FOLDER = './static/uploads'
ALLOWED_EXTENSIONS = {'jpg', 'jpeg', 'png', 'gif'}  # Set the allowed file extensions
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
cors = CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'

app = Flask(__name__)

def hex_to_rgb(hex_color):
    hex_color = hex_color.lstrip('#')
    return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))


def get_colors(image_path, num_colors):
    # Open the image and resize it to speed up processing
    image = Image.open(image_path)
    image = image.resize((200, 200))

    # Get the colors from the image
    colors = image.getcolors(200 * 200)
    colors = sorted(colors, key=lambda t: t[0], reverse=True)

    # Get the dominant colors
    dominant_colors = [colors[0][1]]
    for count, color in colors[1:]:
        if len(dominant_colors) >= num_colors:
            break
        distinct = True
        for dc in dominant_colors:
            distance = math.sqrt(sum([(c1 - c2) ** 2 for c1, c2 in zip(color, dc)]))
            if distance < 50:  # set a threshold for how similar colors can be
                distinct = False
                break
        if distinct:
            dominant_colors.append(color)

    # Remove any similar colors from the dominant color list
    i = 0
    while i < len(dominant_colors):
        color1 = dominant_colors[i]
        j = i + 1
        while j < len(dominant_colors):
            color2 = dominant_colors[j]
            distance = math.sqrt(sum([(c1 - c2) ** 2 for c1, c2 in zip(color1, color2)]))
            if distance < 100:  # set a threshold for how similar colors can be
                dominant_colors.pop(j)
            else:
                j += 1
        i += 1

    # Convert the RGB tuples to hexadecimal values
    dominant_colors_hex = []
    for color in dominant_colors:
        hex_value = '#{:02x}{:02x}{:02x}'.format(*color)
        dominant_colors_hex.append(hex_value)

    # Print the most dominant colors
    # print(f'Top {num_colors} most dominant colors are:')
    # for i, color in enumerate(dominant_colors_hex):
    #     print(f'{i+1}. {color}')

    return dominant_colors_hex

def readImage(img_name):
    img = cv2.imread(img_name)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def resizeAndPad(img, size, pad_color=0):

    h, w = img.shape[:2]
    sh, sw = size

    # interpolation method
    if h > sh or w > sw: # shrinking image
        interp = cv2.INTER_AREA
    else: # stretching image
        interp = cv2.INTER_CUBIC

    # aspect ratio of image
    aspect = w/h  # if on Python 2, you might need to cast as a float: float(w)/h

    # compute scaling and pad sizing
    if aspect > 1: # horizontal image
        new_w = sw
        new_h = np.round(new_w/aspect).astype(int)
        pad_vert = (sh-new_h)/2
        pad_top, pad_bot = np.floor(pad_vert).astype(int), np.ceil(pad_vert).astype(int)
        pad_left, pad_right = 0, 0
    elif aspect < 1: # vertical image
        new_h = sh
        new_w = np.round(new_h*aspect).astype(int)
        pad_horz = (sw-new_w)/2
        pad_left, pad_right = np.floor(pad_horz).astype(int), np.ceil(pad_horz).astype(int)
        pad_top, pad_bot = 0, 0
    else: # square image
        new_h, new_w = sh, sw
        pad_left, pad_right, pad_top, pad_bot = 0, 0, 0, 0

    # set pad color
    if len(img.shape) == 3 and not isinstance(pad_color, (list, tuple, np.ndarray)): # color image but only one color provided
        pad_color = [pad_color]*3

    # scale and pad
    scaled_img = cv2.resize(img, (new_w, new_h), interpolation=interp)
    scaled_img = cv2.copyMakeBorder(scaled_img, pad_top, pad_bot, pad_left, pad_right, borderType=cv2.BORDER_CONSTANT, value=pad_color)

    return scaled_img

def getColoredImage(img, new_color, pattern_image):

    hsv_image = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    h, s, v = cv2.split(hsv_image)
    new_hsv_image = hsv_image

    if new_color is not None:
        color = np.uint8([[new_color]])
        hsv_color = cv2.cvtColor(color, cv2.COLOR_RGB2HSV)
        h.fill(hsv_color[0][0][0])  # todo: optimise to handle black/white walls
        s.fill(hsv_color[0][0][1])
        new_hsv_image = cv2.merge([h, s, v])

    else:
        pattern = cv2.imread('./public/patterns/' + pattern_image)
        hsv_pattern = cv2.cvtColor(pattern, cv2.COLOR_BGR2HSV)
        hp, sp, vp = cv2.split(hsv_pattern)
        # cv2.add(vp, v, vp)
        new_hsv_image = cv2.merge([hp, sp, v])

    new_rgb_image = cv2.cvtColor(new_hsv_image, cv2.COLOR_HSV2RGB)
    return new_rgb_image


def mergeImages(img, colored_image, wall):
    colored_image = cv2.bitwise_and(colored_image, colored_image, mask=wall)
    marked_img = cv2.bitwise_and(img, img, mask=cv2.bitwise_not(wall))
    final_img = cv2.bitwise_xor(colored_image, marked_img)
    return final_img


def saveImage(img_name, img):
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    path_out = f"./static/outputs/{img_name}"
    cv2.imwrite(path_out, img)
    return path_out


def getOutlineImg(img):
    # img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    # img = clahe.apply(img)
    # img = cv2.equalizeHist(img)
    return cv2.Canny(img,50,200)  # todo: can be optimised later

def selectWall(outline_img, position):
    h, w = outline_img.shape[:2]
    
    # Clamp the position coordinates to be within the image bounds
    position = (max(0, min(position[0], w - 1)), max(0, min(position[1], h - 1)))
    
    wall = outline_img.copy()
    scaled_mask = resizeAndPad(outline_img, (h+2, w+2), 255)
    
    # Check dimensions and position before flood fill
    # print("Outline Image Dimensions:", outline_img.shape)
    # print("Position:", position)
    # print("Scaled Mask Dimensions:", scaled_mask.shape)
    
    cv2.floodFill(wall, scaled_mask, position, 255)   # todo: can be optimized later
    cv2.subtract(wall, outline_img, wall) 
    return wall

# Call the function with your image and position
# outline_img = ...  # Load or create your outline image
# position = (x, y)  # Replace with the desired seed point
# selected_wall = selectWall(outline_img, position)








# Call the function with your image and position
# outline_img = ...  # Load or create your outline image
# position = (x, y)  # Replace with the desired seed point
# selected_wall = selectWall(outline_img, position)










def changeColor(image_path, position, new_color, pattern_image):
    start = datetime.timestamp(datetime.now())
    img = readImage(image_path)
    image_name = os.path.basename(image_path)

    colored_image = getColoredImage(img, new_color, pattern_image)

    outline_img = getOutlineImg(img)

    selected_wall = selectWall(outline_img, position)
    
    final_img = mergeImages(img, colored_image, selected_wall)
    
    return saveImage(image_name, final_img)
    # showImages(original_img, colored_image, selected_wall, final_img)



data = []
def recommend_image(image_path):
    dom_color = get_colors(image_path, 20)
    a = dom_color[:5]
    while(len(a)<6):
        a.append('')
        data.append(a)
    color_palette_data = []
    for palette in data:
        color_dict = {
            'color1': palette[0],
            'color2': palette[1],
            'color3': palette[2],
            'color4': palette[3],
        }
        color_palette_data.append(color_dict)

    # Convert the list of dictionaries into a Pandas DataFrame
    df = pd.DataFrame(color_palette_data)

    for column in df.columns:
        df[column] = df[column].apply(hex_to_rgb)

    df['color1_red'] = df['color1'].apply(lambda x: x[0])
    df['color1_green'] = df['color1'].apply(lambda x: x[1])
    df['color1_blue'] = df['color1'].apply(lambda x: x[2])

    df['color2_red'] = df['color2'].apply(lambda x: x[0])
    df['color2_green'] = df['color2'].apply(lambda x: x[1])
    df['color2_blue'] = df['color2'].apply(lambda x: x[2])

    df['color3_red'] = df['color3'].apply(lambda x: x[0])
    df['color3_green'] = df['color3'].apply(lambda x: x[1])
    df['color3_blue'] = df['color3'].apply(lambda x: x[2])

    df['color4_red'] = df['color4'].apply(lambda x: x[0])
    df['color4_green'] = df['color4'].apply(lambda x: x[1])
    df['color4_blue'] = df['color4'].apply(lambda x: x[2])



    # Drop the original RGB columns
    df = df.drop(['color1', 'color2', 'color3','color4'], axis=1)

    X = df[['color2_red', 'color2_green', 'color2_blue',
        'color3_red', 'color3_green', 'color3_blue','color4_red','color4_green','color4_blue']]
    
    y_pred = model.predict(X)
    print(y_pred)
    return changeColor(image_path, (300, 100), [y_pred[len(y_pred)-1][0], y_pred[len(y_pred)-1][1], y_pred[len(y_pred)-1][2]], None)
    

@app.route('/recommend', methods=['POST'])
@cross_origin()
def get_data():
    if len(request.files) ==0:
        return "File empty", 400
    file = request.files['file']
    if file.filename == '':
        return "No image selected for uploading", 400

    if file and allowed_file(file.filename):
        try:
            filename = secure_filename(file.filename)
            file.save(os.path.join(UPLOAD_FOLDER, filename))
            path = os.path.join(UPLOAD_FOLDER, filename)
            ans = recommend_image(path)
            return jsonify({"prediction": ans}), 200
        except requests.exceptions.RequestException as e:
            return jsonify({'error': 'Failed to fetch data from the external API'}), 500
    else:
        return "Invalid file extension. Only JPG, JPEG, PNG, and GIF are allowed.", 400

if __name__ == '__main__':
    app.run(host='localhost',debug=True, port=5001)