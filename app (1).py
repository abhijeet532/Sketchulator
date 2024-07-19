from keras.models import load_model
model=load_model('num_detect (1).h5')
import  numpy as np
import cv2
from keras.preprocessing import image
import matplotlib.pyplot as plt
def mnist_compatible(image_path, target_size=(28, 28)):

    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    plt.imshow(img)
    plt.show()



    img_resized = cv2.resize(img, target_size)


    img_inverted = 255 - img_resized


    img_normalized = img_inverted.astype('float32') / 255.0


    img_array = image.img_to_array(img_normalized)

    img_reshaped = img_array.reshape((*target_size, 1))

    return img_reshaped

def recognize_digit(dict):
    path1 = dict['composite']
    arr = mnist_compatible(path1)

    arr = np.expand_dims(arr, axis=0)
    prediction = np.argmax(model.predict(arr))
    return prediction

def calculator(image1, operation, image2):
    # Recognize the drawn digits
    num1 = recognize_digit(image1)
    num2 = recognize_digit(image2)

    # Perform the calculator operation
    if operation == "add":
        result = num1 + num2
    elif operation == "subtract":
        result = num1 - num2
    elif operation == "multiply":
        result = num1 * num2
    elif operation == "divide":
        result = num1 / num2

    return result
import gradio as gr

with gr.Blocks() as demo:
    with gr.Row():
        with gr.Column():
            image1 = gr.Paint(label="Draw First Number", type="filepath", brush=gr.Brush(colors=["#32cc70"]), canvas_size=(301, 301))
            operation = gr.Radio(["add", "subtract", "multiply", "divide"])
            image2 = gr.Paint(label="Draw Second Number", type="filepath", brush=gr.Brush(colors=["#32cc70"]), canvas_size=(301, 301))
            submit_btn = gr.Button(value="Calculate")
        with gr.Column():
            result = gr.Textbox(label="Result")

    submit_btn.click(calculator, inputs=[image1, operation, image2], outputs=[result])
demo.launch(share='True')