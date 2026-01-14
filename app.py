import os
import numpy as np
import torch
from flask import Flask, render_template, request
from werkzeug.utils import secure_filename
from PIL import Image
from torchvision import transforms
from torchvision.models import efficientnet_b0

model = efficientnet_b0(pretrained=False)

num_classes = 15
model.classifier[1] = torch.nn.Linear(model.classifier[1].in_features, num_classes)

model.load_state_dict(torch.load('efficientnet_model.pth', map_location=torch.device('cpu')))
model.eval()  

app = Flask(__name__)

UPLOAD_FOLDER = 'static/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif','jfif'}

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

class_labels = [
    'Pepper__bell___Bacterial_spot', 'Pepper__bell___healthy',
    'Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy',
    'Tomato_Bacterial_spot', 'Tomato_Early_blight', 'Tomato_Late_blight',
    'Tomato_Leaf_Mold', 'Tomato_Septoria_leaf_spot',
    'Tomato_Spider_mites_Two_spotted_spider_mite', 'Tomato__Target_Spot',
    'Tomato__Tomato_YellowLeaf__Curl_Virus', 'Tomato__Tomato_mosaic_virus',
    'Tomato_healthy'
]

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def preprocess_image(image_path):
    img = Image.open(image_path).convert('RGB')
    img = img.resize((224, 224))
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    img_tensor = transform(img)
    img_tensor = img_tensor.unsqueeze(0)
    return img_tensor

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction_result = None
    uploaded_image = None
    error_message = None
    
    if request.method == 'POST':
        if 'file' not in request.files:
            error_message = 'No file part'
        else:
            file = request.files['file']
            if file.filename == '':
                error_message = 'No selected file'
            elif file and allowed_file(file.filename):
                filename = secure_filename(file.filename)
                filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                file.save(filepath)
                uploaded_image = filename
                
                try:
                    img_tensor = preprocess_image(filepath)
                    
                    with torch.no_grad():
                        outputs = model(img_tensor)
                        probabilities = torch.nn.functional.softmax(outputs, dim=1)
                        confidence, predicted_idx = torch.max(probabilities, 1)
                    
                    predicted_class_idx = predicted_idx.item()
                    predicted_class_name = class_labels[predicted_class_idx]
                    confidence_value = confidence.item() * 100
                    
                    top_probs, top_indices = torch.topk(probabilities, 3, dim=1)
                    top_probs = top_probs.squeeze().tolist()
                    top_indices = top_indices.squeeze().tolist()
                    
                    top_predictions = []
                    for i, idx in enumerate(top_indices):
                        top_predictions.append({
                            'class': class_labels[idx],
                            'confidence': f"{top_probs[i] * 100:.2f}%"
                        })
                    
                    prediction_result = {
                        'main': {
                            'class': predicted_class_name,
                            'confidence': f"{confidence_value:.2f}%"
                        },
                        'top3': top_predictions
                    }
                except Exception as e:
                    error_message = f"Error processing image: {str(e)}"
    
    return render_template('index.html', 
                          prediction=prediction_result,
                          image=uploaded_image,
                          error=error_message)

if __name__ == "__main__":
    app.run(debug=True)