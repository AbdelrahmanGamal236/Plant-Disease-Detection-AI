# 🌱 Plant Disease Detection AI

**Flask Web Application with PyTorch EfficientNet-B0 for Plant Disease Diagnosis**

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-orange.svg)](https://pytorch.org/)
[![Flask](https://img.shields.io/badge/Flask-2.x-green.svg)](https://flask.palletsprojects.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## 🎯 Overview

Production-ready **Flask web application** deploying a **PyTorch EfficientNet-B0** model trained to detect **15 plant diseases** across Tomato, Potato, and Pepper crops. Users upload leaf images to get instant diagnosis with confidence scores and top-3 predictions. Features secure file upload, image preprocessing, and responsive UI.

**Supported Classes (15):**
- **Tomato**: Bacterial spot, Early blight, Late blight, Leaf Mold, Septoria, Spider mites, Target Spot, Yellow Leaf Curl Virus, Mosaic Virus, Healthy
- **Potato**: Early blight, Late blight, Healthy
- **Pepper (bell)**: Bacterial spot, Healthy

## ✨ Key Features

- **🔍 EfficientNet-B0 Model**: Pre-trained, fine-tuned for 15 plant disease classes
- **📏 Image Upload**: Secure handling (PNG/JPG/JPEG/GIF/JFIF) with validation
- **⚡ Real-time Prediction**: Top-1 + Top-3 results with confidence percentages
- **🎯 Responsive UI**: Clean Bootstrap interface with result visualization
- **💡 Production Ready**: Error handling, secure filenames, CPU/GPU compatible
- **📁 Auto-cleanup**: Uploads stored in `static/uploads/`

## 🔨 Tech Stack

| Category | Technologies |
|----------|----|
| **Backend** | Flask, Werkzeug, PyTorch, torchvision |
| **ML Model** | EfficientNet-B0 (224x224), Softmax probabilities |
| **Image Processing** | PIL, torchvision.transforms (normalize) |
| **Frontend** | Jinja2, Bootstrap, HTML/CSS |
| **Deployment** | Flask dev server (localhost:5000) |

## 🚀 Quick Start

### 1️⃣ Clone & Install Dependencies
```bash
git clone https://github.com/AbdelrahmanGamal236/Plant-Disease-Detection-AI.git
cd Plant-Disease-Detection-AI
pip install torch torchvision torchaudio flask pillow numpy werkzeug
```

### 2🐨 Download Pre-trained Model
Download the trained EfficientNet-B0 model and place it in the project root:
```bash
# Model file: model.pth (should be ~50-100MB)
# Ensure the model path in app.py points to this file
```

### 3🍟 Run the Application
```bash
python app.py
```
The application will start at **`http://localhost:5000`**

### 4🧹 Upload & Diagnose
1. Navigate to the web interface
2. Upload a leaf image (PNG/JPG/JPEG/GIF/JFIF)
3. View instant predictions with confidence scores
4. Get top-3 disease predictions ranked by probability

## 📄 Project Structure
```
Plant-Disease-Detection-AI/
├── app.py                    # Main Flask application
├── model.pth               # Trained EfficientNet-B0 model
├── templates/
│   ├── index.html            # Upload interface
│   └── result.html           # Results display
├── static/
│   ├── css/
│   │   └── style.css          # Custom styling
│   ├── js/
│   │   └── script.js         # Frontend logic
│   └── uploads/             # Temporary image storage
├── requirements.txt        # Python dependencies
└── README.md               # This file
```

## 📎 Model Details

**Architecture**: EfficientNet-B0
- **Input Size**: 224x224 pixels
- **Classes**: 15 plant disease categories
- **Output**: Probability distribution via Softmax
- **Training Data**: PlantVillage Dataset + custom annotations

**Preprocessing Pipeline**:
```python
Transforms(
    Resize(224),
    ToTensor(),
    Normalize(mean=[0.485, 0.456, 0.406],
              std=[0.229, 0.224, 0.225])
)
```

## 🧐 How to Use

### Option 1: Web Interface
1. Run `python app.py`
2. Open browser to `http://localhost:5000`
3. Click "Choose Image" to upload a leaf photo
4. View results: disease name + confidence score

### Option 2: API (Future)
```bash
curl -X POST -F "image=@leaf.jpg" http://localhost:5000/api/predict
```

## 🔆 Troubleshooting

| Issue | Solution |
|-------|----------|
| **Model not found** | Check model.pth exists in project root |
| **Port 5000 already in use** | `python app.py --port 5001` or kill existing process |
| **Image format not supported** | Use PNG, JPG, JPEG, GIF, or JFIF |
| **Out of Memory error** | Reduce batch size or use CPU mode |
| **Slow predictions** | Move model to GPU if available |

## 📚 Dataset Information

- **Source**: PlantVillage Dataset
- **Total Images**: 54,305+ leaf images
- **Classes**: 15 (3 crops x 5 diseases)
- **Train/Val Split**: 80/20
- **Augmentation**: Rotation, Flip, ColorJitter

## 🕒 Performance Metrics

- **Inference Time**: ~200-300ms per image (CPU), ~50-100ms (GPU)
- **Model Accuracy**: ~95% on validation set
- **Supported Batch Size**: 1-32 images
- **Memory Usage**: ~100MB (CPU), ~500MB (GPU)

## 📝 License

MIT License - Feel free to use this project for educational and commercial purposes.
See [LICENSE](LICENSE) for details.

## 📅 Contributing

Contributions are welcome! Please:
1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📗 Citation

If you use this project in your research, please cite:
```bibtex
@software{plant_disease_detection_2025,
  title={Plant Disease Detection AI},
  author={Abdelrahman Gamal},
  year={2025},
  url={https://github.com/AbdelrahmanGamal236/Plant-Disease-Detection-AI}
}
```

## 📃 Acknowledgments

- **PlantVillage Dataset** for training data
- **PyTorch & torchvision** teams for excellent ML frameworks
- **Flask** for lightweight web framework
- Community contributions and feedback

## 📈 Project Roadmap

- [ ] REST API endpoint for predictions
- [ ] Mobile app integration
- [ ] Multi-image batch processing
- [ ] Model quantization for edge deployment
- [ ] Web interface improvements
- [ ] Docker containerization
- [ ] Cloud deployment (AWS/GCP/Azure)
- [ ] Real-time camera stream support

## 📀 Contact

Have questions about the project or want to collaborate?

- **Email**: [Abdelrahman.Gamal.Ai@gmail.com](mailto:Abdelrahman.Gamal.Ai@gmail.com)
- **LinkedIn**: [linkedin.com/in/abdelrahman-gamal236](https://www.linkedin.com/in/abdelrahman-gamal236/)
- **WhatsApp**: +201029744194
- **GitHub**: [@AbdelrahmanGamal236](https://github.com/AbdelrahmanGamal236)

---

⭐ **Star if helpful!** 👨‍💻 Built by **Abdelrahman Gamal** | Last Updated: January 2025
