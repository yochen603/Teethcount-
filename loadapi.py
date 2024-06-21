import torch
from torchvision.models import efficientnet_b0,resnext50_32x4d,mobilenet_v2
from PIL import Image
from torchvision import transforms
import numpy as np
import matplotlib.pyplot as plt
import io
from stl import mesh

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def xyplane(filename):
    your_mesh = mesh.Mesh.from_file(filename)
    Zmin = np.min(your_mesh.vectors[:, :, 2])
    Zmax = np.max(your_mesh.vectors[:, :, 2])
    print(f"Minimum Z value in the mesh is {Zmin}, and maximum Z value is {Zmax}")

    num_planes = 15  # Number of XY planes to plot
    z_planes = np.linspace(Zmin + (Zmax - Zmin) * 0.30, Zmin + (Zmax - Zmin) * 0.70, num_planes)

    all_points = []

    for z_plane in z_planes:
        i_facets = 0
        points = []
        for i in range(your_mesh.vectors.shape[0]):
            facet = your_mesh.vectors[i, :, :]
            z_coords = facet[:, 2]
            if np.min(z_coords) <= z_plane <= np.max(z_coords):
                i_facets += 1
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
        
def xyplane_fast(filename):
    your_mesh = mesh.Mesh.from_file(filename)

    Zmin = np.min(your_mesh.z)
    Zmax = np.max(your_mesh.z)
    print(f"Minimum Z value in the mesh is {Zmin}, and maximum Z value is {Zmax}")

    num_planes = 20  # Number of XY planes to plot
    alpha = 0.5
    beta = 0.6
    z_planes = np.linspace(Zmin + (Zmax - Zmin) * alpha, Zmin + (Zmax - Zmin) * beta, num_planes)

    fig, ax = plt.subplots(figsize=(8, 8))

    for z_plane in z_planes:
        # Get the vertices of the mesh
        vertices = your_mesh.vectors.reshape(-1, 3)

        # Filter vertices based on the Z-plane
        mask = (vertices[:, 2] >= z_plane - 0.01) & (vertices[:, 2] <= z_plane + 0.01)
        selected_vertices = vertices[mask]

        # Plot the selected vertices
        ax.scatter(selected_vertices[:, 0], selected_vertices[:, 1], s=5, zorder=10, color='blue', linewidths=0.5)

    ax.set_aspect('equal')
    ax.axis('off')  # Remove the axes
    plt.tight_layout()

    # Save the plot to a bytes buffer
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0, dpi=300)
    plt.close(fig)
    buf.seek(0)

    # Create a PIL Image from the bytes buffer
    image = Image.open(buf)
    image = image.convert('RGB')

    print("XY plane plot generated")
    return image


# Initialize the EfficientNet model architecture
def model1(stl_file_path,input_image):
    model = efficientnet_b0()

    # Modify the classifier layer to match the number of classes in your task
    num_classes = 16  # Replace with the number of classes in your classification task
    model.classifier[1] = torch.nn.Linear(model.classifier[1].in_features, num_classes)

    # Load the saved model state dictionary
    state_dict = torch.load('efficientnet_size512_epoch25_0.0001.pth')

    # Load the state dictionary into the model
    model.load_state_dict(state_dict)

    # Set the model to evaluation mode
    model.eval()

    # Define the data transformations for inference
    inference_transforms = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor(),
    ])

    # Ask the user for the path to the STL file
    #stl_file_path = input("Enter the path to the STL file: ")
    
    # Generate the XY plane plot and get the image
    #input_image = xyplane(stl_file_path)
    
    if input_image:
        # Preprocess the input image
        input_tensor = inference_transforms(input_image)
        input_batch = input_tensor.unsqueeze(0)

        # Move the input batch to the same device as the model
        input_batch = input_batch.to(device)
        model.to(device)
    
        # Perform inference
        with torch.no_grad():
            outputs = model(input_batch)
    
        # Get the predicted class probabilities
        probabilities = torch.softmax(outputs, dim=1)

        # Get the predicted class index
        _, predicted_index = torch.max(probabilities, 1)
        #print(predicted_index)

        # Map the predicted index to the corresponding class label
        class_labels = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '16']  # Replace with your class labels
        predicted_label = class_labels[predicted_index.item()]
    
        #print('Model1 Predicted label:', predicted_label)
    else:
        print("Failed to generate the XY plane plot.")
    return predicted_label

# Initialize the resNext model architecture
def model2(stl_file_path,input_image):
    model = resnext50_32x4d()

    # Modify the classifier layer to match the number of classes in your task
    num_classes = 16  # Replace with the number of classes in your classification task
    model.fc = torch.nn.Linear(model.fc.in_features, num_classes)

    # Load the saved model state dictionary
    state_dict = torch.load('model2.pth')

    # Load the state dictionary into the model
    model.load_state_dict(state_dict)

    # Set the model to evaluation mode
    model.eval()

    # Define the data transformations for inference
    inference_transforms = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor(),
    ])

    # Ask the user for the path to the STL file
    #stl_file_path = input("Enter the path to the STL file: ")
    
    # Generate the XY plane plot and get the image
    #input_image = xyplane(stl_file_path)
    
    if input_image:
        # Preprocess the input image
        input_tensor = inference_transforms(input_image)
        input_batch = input_tensor.unsqueeze(0)

        # Move the input batch to the same device as the model
        input_batch = input_batch.to(device)
        model.to(device)
    
        # Perform inference
        with torch.no_grad():
            outputs = model(input_batch)
    
        # Get the predicted class probabilities
        probabilities = torch.softmax(outputs, dim=1)

        # Get the predicted class index
        _, predicted_index = torch.max(probabilities, 1)
        #print(predicted_index)

        # Map the predicted index to the corresponding class label
        class_labels = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '16']  # Replace with your class labels
        predicted_label = class_labels[predicted_index.item()]
    
        #print('Model2 Predicted label:', predicted_label)
    else:
        print("Failed to generate the XY plane plot.")
    return predicted_label

# Initialize the mobilenet model architecture
def model3(stl_file_path,input_image):
    model = mobilenet_v2()

    # Modify the classifier layer to match the number of classes in your task
    num_classes = 16  # Replace with the number of classes in your classification task
    model.classifier[1] = torch.nn.Linear(model.classifier[1].in_features, num_classes)

    # Load the saved model state dictionary
    state_dict = torch.load('model3.pth')

    # Load the state dictionary into the model
    model.load_state_dict(state_dict)

    # Set the model to evaluation mode
    model.eval()

    # Define the data transformations for inference
    inference_transforms = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor(),
    ])

    # Ask the user for the path to the STL file
    #stl_file_path = input("Enter the path to the STL file: ")
    
    # Generate the XY plane plot and get the image
    #input_image = xyplane(stl_file_path)
    
    if input_image:
        # Preprocess the input image
        input_tensor = inference_transforms(input_image)
        input_batch = input_tensor.unsqueeze(0)

        # Move the input batch to the same device as the model
        input_batch = input_batch.to(device)
        model.to(device)
    
        # Perform inference
        with torch.no_grad():
            outputs = model(input_batch)
    
        # Get the predicted class probabilities
        probabilities = torch.softmax(outputs, dim=1)

        # Get the predicted class index
        _, predicted_index = torch.max(probabilities, 1)
        #print(predicted_index)

        # Map the predicted index to the corresponding class label
        class_labels = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '16']  # Replace with your class labels
        predicted_label = class_labels[predicted_index.item()]
    
        #print('Model3 Predicted label:', predicted_label)
    else:
        print("Failed to generate the XY plane plot.")
    return predicted_label

stl_file_path=input("Enter the path to the STL file: ")
input_image= xyplane(stl_file_path)
label1=model1(stl_file_path,input_image)
label2=model2(stl_file_path,input_image)
label3=model2(stl_file_path,input_image)
if int(label2)<=5 and int(label3)<=5:
    print("label:",label2)
else:
    print("label:",label1)

#if label1==label2:
#    print('Predicted label:', label1)
#elif label1==label3 or label2==label3:
#    print('Predicted label:',label3)
#else:
#    print('Predicted label:',label1)

