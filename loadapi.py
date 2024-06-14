import torch
from torchvision.models import efficientnet_b0, resnext50_32x4d, mobilenet_v2
from PIL import Image
from torchvision import transforms
import numpy as np
import matplotlib.pyplot as plt
import io
from stl import mesh
import time

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def xyplane(filename):
    your_mesh = mesh.Mesh.from_file(filename)
    Zmin = np.min(your_mesh.vectors[:, :, 2])
    Zmax = np.max(your_mesh.vectors[:, :, 2])
    print(f"Minimum Z value in the mesh is {Zmin}, and maximum Z value is {Zmax}")

    num_planes = 20  # Number of XY planes to plot
    z_planes = np.linspace(Zmin + (Zmax - Zmin) * 0.40, Zmin + (Zmax - Zmin) * 0.60, num_planes)

    all_points = []

    for z_plane in z_planes:
        points = []
        for facet in your_mesh.vectors:
            z_coords = facet[:, 2]
            if np.min(z_coords) <= z_plane <= np.max(z_coords):
                for j in range(3):
                    if (facet[j, 2] <= z_plane <= facet[(j + 1) % 3, 2]) or (
                        facet[(j + 1) % 3, 2] <= z_plane <= facet[j, 2]):
                        t = (z_plane - facet[j, 2]) / (facet[(j + 1) % 3, 2] - facet[j, 2])
                        point = facet[j, :] + t * (facet[(j + 1) % 3, :] - facet[j, :])
                        points.append(point)

        points = np.array(points)
        if len(points) > 0:
            points = points[:, :2]  # Extract only the x and y coordinates
            all_points.extend(points)

    all_points = np.array(all_points)

    if len(all_points) > 0:
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.scatter(all_points[:, 0], all_points[:, 1], s=5)
        ax.axis('off')  # Remove the axes
        plt.tight_layout()

        # Save the plot to a bytes buffer
        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0)
        print("Image into buffer")
        buf.seek(0)

        # Create a PIL Image from the bytes buffer
        image = Image.open(buf)
        image = image.convert('RGB') 
        return image
    else:
        print("No points found for the specified XY planes.")
        return None

def load_model(model_name, num_classes, model_path):
    start_time = time.time()  # 开始时间
    if model_name == 'efficientnet_b0':
        model = efficientnet_b0()
        model.classifier[1] = torch.nn.Linear(model.classifier[1].in_features, num_classes)
    elif model_name == 'resnext50_32x4d':
        model = resnext50_32x4d()
        model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
    elif model_name == 'mobilenet_v2':
        model = mobilenet_v2()
        model.classifier[1] = torch.nn.Linear(model.classifier[1].in_features, num_classes)

    map_location = torch.device('cpu') if not torch.cuda.is_available() else None
    state_dict = torch.load(model_path, map_location=map_location)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    load_time = time.time() - start_time  # 计算加载时间
    return model,load_time

def predict(model, input_image):
    inference_transforms = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor(),
    ])
    
    if input_image:
        start_time = time.time()  # 开始时间

        input_tensor = inference_transforms(input_image)
        input_batch = input_tensor.unsqueeze(0).to(device)
    
        with torch.no_grad():
            outputs = model(input_batch)
    
        probabilities = torch.softmax(outputs, dim=1)
        _, predicted_index = torch.max(probabilities, 1)

        class_labels = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '16']  
        predicted_label = class_labels[predicted_index.item()]

        predict_time = time.time() - start_time  # 计算预测时间
        return predicted_label,predict_time
    else:
        print("Failed to generate the XY plane plot.")
        return None

# def model1(stl_file_path, input_image):
#     model = load_model('efficientnet_b0', 16, 'model1.pth')
#     return predict(model, input_image)

# def model2(stl_file_path, input_image):
#     model = load_model('resnext50_32x4d', 16, 'model2.pth')
#     return predict(model, input_image)

# def model3(stl_file_path, input_image):
#     model = load_model('mobilenet_v2', 16, 'model3.pth')
#     return predict(model, input_image)

# def get_final_prediction(stl_file_path, input_image):
#     label1 = model1(stl_file_path, input_image)
#     label2 = model2(stl_file_path, input_image)
#     label3 = model3(stl_file_path, input_image)
    
#     if label1 == label2:
#         return label1
#     elif label1 == label3 or label2 == label3:
#         return label3
#     else:
#         return label1

def get_final_prediction(stl_file_path, input_image):
    # 加载模型并记录时间
    model1, load_time1 = load_model('efficientnet_b0', 16, 'model1.pth')
    model2, load_time2 = load_model('resnext50_32x4d', 16, 'model2.pth')
    model3, load_time3 = load_model('mobilenet_v2', 16, 'model3.pth')

    # 预测并记录时间
    label1, predict_time1 = predict(model1, input_image)
    label2, predict_time2 = predict(model2, input_image)
    label3, predict_time3 = predict(model3, input_image)

    # 选择预测结果
    final_label = label1
    if label1 == label2:
        final_label = label1
    elif label1 == label3 or label2 == label3:
        final_label = label3
    else:
        final_label = label1

    return final_label, (load_time1 + load_time2 + load_time3), (predict_time1 + predict_time2 + predict_time3)