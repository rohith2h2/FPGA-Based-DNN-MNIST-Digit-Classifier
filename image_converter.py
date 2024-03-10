from PIL import Image
import numpy as np

pic_name = ["pic_0.bmp",
            "pic_1.bmp",
            "pic_2.bmp",
            "pic_3.bmp",
            "pic_4.bmp",
            "pic_5.bmp",
            "pic_6.bmp",
            "pic_7.bmp",
            "pic_8.bmp",
            "pic_9.bmp"]

save_pic_name = ["input_0",
                 "input_1",
                 "input_2",
                 "input_3",
                 "input_4",
                 "input_5",
                 "input_6",
                 "input_7",
                 "input_8",
                 "input_9"]

for i in range(10):
    test_img = Image.open("/Users/hr/my_files/test/test_" + str(i) + ".bmp")

    # Convert image to array
    test_img_array = np.array(test_img).astype('float32') / 255
    test_img_array = test_img_array.reshape((1, 784))

    # Save image parameters
    with open(save_pic_name[i] + ".h", 'w') as f:
        weights_str = ', '.join(map(str, test_img_array.flatten()))
        f.write("float " + save_pic_name[i] + "[784]={" + weights_str + "};\n")

    print("Image parameters saved for", save_pic_name[i])
