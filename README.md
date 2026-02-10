# 🌿 Pestector - Plant Disease Detection System

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Node.js](https://img.shields.io/badge/Node.js-14+-green.svg)](https://nodejs.org/)
[![Deep Learning](https://img.shields.io/badge/Deep%20Learning-TensorFlow-orange.svg)](https://www.tensorflow.org/)

An intelligent plant disease detection platform that leverages deep learning to identify plant diseases from leaf images. Built with a modern two-backend architecture for scalability and maintainability.

---

## 📋 Table of Contents

- [Overview](#overview)
- [System Architecture](#system-architecture)
- [Features](#features)
- [Technology Stack](#technology-stack)
- [Dataset Information](#dataset-information)
- [Installation](#installation)
- [Usage](#usage)
- [API Documentation](#api-documentation)
- [Project Structure](#project-structure)
- [Model Performance](#model-performance)
- [Contributing](#contributing)
- [License](#license)
- [Acknowledgments](#acknowledgments)

---

## 🌟 Overview

Pestector is a cutting-edge plant disease detection system designed to help farmers, agricultural professionals, and gardening enthusiasts identify plant diseases quickly and accurately. By simply uploading an image of a plant leaf, users receive instant diagnosis powered by state-of-the-art deep learning models.

### Key Highlights

- **Real-time Disease Detection**: Instant analysis of plant leaf images
- **38 Disease Classes**: Covers a wide range of crop diseases
- **87,000+ Training Images**: Trained on a comprehensive dataset
- **Two-Backend Architecture**: Separation of concerns for better scalability
- **User-Friendly Interface**: Clean, responsive web interface

---

## 🏗️ System Architecture

Pestector implements a **Two-Backend Architecture** to separate AI processing from application logic, enhancing scalability and maintainability.

```
┌─────────────────────────────────────────────────────────────┐
│                        Frontend Layer                        │
│           (Vanilla JS + HTML + Tailwind CSS)                │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────┐
│                   Node.js Backend Server                     │
│          (Application Logic & API Management)               │
│  • User Authentication & Authorization                      │
│  • Request Routing                                          │
│  • Database Management                                      │
│  • Static File Serving (from public/)                      │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────┐
│                   Python AI Backend Server                   │
│              (Deep Learning & Image Processing)             │
│  • Image Preprocessing                                      │
│  • Deep Learning Model Inference                            │
│  • Disease Classification                                   │
│  • Prediction Results Generation                            │
└─────────────────────────────────────────────────────────────┘
```

### Data Flow

1. **User uploads** plant leaf image via web interface
2. **Frontend** sends image to Node.js backend
3. **Node.js backend** forwards image to Python AI backend
4. **Python AI backend** processes image and runs ML model
5. **Classification results** return to Node.js backend
6. **Node.js backend** stores results in database
7. **Results displayed** to user in real-time

### Repository Structure

- **Python AI Backend**: [github.com/Abdelrahman968/aibackend-pestector](https://github.com/Abdelrahman968/aibackend-pestector)
- **Node.js Backend**: [github.com/Abdelrahman968/pestector-nodeJS](https://github.com/Abdelrahman968/pestector-nodeJS)

---

## ✨ Features

### Core Functionality

- ✅ **Image Upload**: Support for common image formats (JPG, PNG, JPEG)
- ✅ **Real-time Analysis**: Instant disease detection and classification
- ✅ **38 Disease Categories**: Comprehensive coverage of plant diseases
- ✅ **Confidence Scores**: Prediction confidence for each classification
- ✅ **User Management**: Secure authentication and user profiles
- ✅ **History Tracking**: View past disease detections
- ✅ **Responsive Design**: Works on desktop, tablet, and mobile

### Advanced Features

- 🔒 **Secure Authentication**: JWT-based user authentication
- 📊 **Dashboard**: User analytics and detection history
- 🎨 **Modern UI**: Clean interface built with Tailwind CSS
- 🚀 **RESTful API**: Well-documented API endpoints
- 📱 **Mobile Responsive**: Optimized for all screen sizes

---

## 🛠️ Technology Stack

### Frontend

- **JavaScript**: Vanilla JS for lightweight performance
- **HTML5**: Semantic markup
- **Tailwind CSS**: Utility-first CSS framework
- **Fetch API**: For HTTP requests

### Node.js Backend

- **Runtime**: Node.js 14+
- **Framework**: Express.js
- **Database**: MongoDB / PostgreSQL
- **Authentication**: JWT (JSON Web Tokens)
- **File Upload**: Multer
- **HTTP Client**: Axios

### Python AI Backend

- **Language**: Python 3.8+
- **Deep Learning**: TensorFlow / Keras / PyTorch
- **Image Processing**: OpenCV, PIL
- **Web Framework**: Flask / FastAPI
- **Data Processing**: NumPy, Pandas

### DevOps

- **Version Control**: Git & GitHub
- **Containerization**: Docker (optional)
- **API Testing**: Postman

---

## 📊 Dataset Information

The AI model is trained on the **New Plant Diseases Dataset** from Kaggle.

### Dataset Details

- **Source**: [New Plant Diseases Dataset](https://www.kaggle.com/datasets/vipoooool/new-plant-diseases-dataset)
- **Total Images**: ~87,000 RGB images
- **Image Categories**: 38 different classes
- **Image Types**: Healthy and diseased crop leaves
- **Augmentation**: Offline data augmentation applied

### Dataset Split

```
Training Set   : 80% (~70,000 images)
Validation Set : 20% (~17,000 images)
Test Set       : 33 images (separate test folder)
```

### Supported Plant Categories

The dataset covers various crops including:

- 🍎 Apple (4 classes: healthy, apple scab, black rot, cedar rust)
- 🌽 Corn (4 classes: healthy, cercospora, common rust, northern leaf blight)
- 🍇 Grape (4 classes: healthy, black rot, esca, leaf blight)
- 🍑 Peach (2 classes: healthy, bacterial spot)
- 🌶️ Pepper (2 classes: healthy, bacterial spot)
- 🥔 Potato (3 classes: healthy, early blight, late blight)
- 🍓 Strawberry (2 classes: healthy, leaf scorch)
- 🍅 Tomato (10 classes: healthy, various diseases)
- And more...

---

## 🚀 Installation

### Prerequisites

- Node.js 14+ and npm
- Python 3.8+
- MongoDB or PostgreSQL
- Git

### Clone Repositories

```bash
# Clone Node.js Backend
git clone https://github.com/Abdelrahman968/pestector-nodeJS.git
cd pestector-nodeJS

# Clone Python AI Backend
git clone https://github.com/Abdelrahman968/aibackend-pestector.git
cd aibackend-pestector
```

### Setup Node.js Backend

```bash
cd pestector-nodeJS

# Install dependencies
npm install

# Create .env file
cp .env.example .env

# Configure environment variables
# Edit .env with your database credentials, JWT secret, etc.

# Start the server
npm start
```

### Setup Python AI Backend

```bash
cd aibackend-pestector

# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Download the trained model (if not included)
# Place model file in /models directory

# Start the server
python app.py
```

### Configuration

#### Node.js Backend (.env)

```env
PORT=3000
MONGODB_URI=mongodb://localhost:27017/pestector
JWT_SECRET=your_secret_key_here
AI_BACKEND_URL=http://localhost:5000
```

#### Python AI Backend (config.py)

```python
PORT = 5000
MODEL_PATH = './models/plant_disease_model.h5'
IMAGE_SIZE = (224, 224)
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
```

---

## 💻 Usage

### Starting the Application

1. **Start Python AI Backend** (Terminal 1):
   ```bash
   cd aibackend-pestector
   python app.py
   ```

2. **Start Node.js Backend** (Terminal 2):
   ```bash
   cd pestector-nodeJS
   npm start
   ```

3. **Access the Application**:
   Open your browser and navigate to `http://localhost:3000`

### Using the Web Interface

1. **Register/Login**: Create an account or log in
2. **Upload Image**: Click "Upload" and select a plant leaf image
3. **View Results**: See the disease prediction with confidence score
4. **Check History**: View past detections in your dashboard

---

## 📡 API Documentation

### Node.js Backend Endpoints

#### Authentication

```http
POST /api/auth/register
POST /api/auth/login
POST /api/auth/logout
GET  /api/auth/me
```

#### Disease Detection

```http
POST /api/detect
GET  /api/detections
GET  /api/detections/:id
DELETE /api/detections/:id
```

#### User Management

```http
GET  /api/users/profile
PUT  /api/users/profile
GET  /api/users/history
```

### Python AI Backend Endpoints

#### Prediction

```http
POST /predict
```

**Request Body** (multipart/form-data):
```json
{
  "file": "<image_file>"
}
```

**Response**:
```json
{
  "success": true,
  "prediction": {
    "class": "Tomato___Late_blight",
    "confidence": 0.95,
    "disease_name": "Late Blight",
    "plant_type": "Tomato"
  },
  "timestamp": "2026-02-04T10:30:00Z"
}
```

#### Health Check

```http
GET /health
```

**Response**:
```json
{
  "status": "healthy",
  "model_loaded": true,
  "version": "1.0.0"
}
```

---

## 📁 Project Structure

### Node.js Backend

```
├── 📁 config
│   ├── 📄 config.js
│   └── 📄 index.js
├── 📁 controllers
│   ├── 📄 index.js
│   └── 📄 recommendationController.js
├── 📁 middleware
│   ├── 📄 auth.js
│   ├── 📄 guest.js
│   ├── 📄 index.js
│   └── 📄 isAdmin.js
├── 📁 models
│   ├── 📄 Analytics.js
│   ├── 📄 AuditLog.js
│   ├── 📄 Chat.js
│   ├── 📄 Comment.js
│   ├── 📄 Contact.js
│   ├── 📄 GuestUser.js
│   ├── 📄 History.js
│   ├── 📄 ModelFeedback.js
│   ├── 📄 Notification.js
│   ├── 📄 Plant.js
│   ├── 📄 Post.js
│   ├── 📄 Recommendation.js
│   ├── 📄 Reminder.js
│   ├── 📄 Subscription.js
│   ├── 📄 TreatmentPlan.js
│   ├── 📄 TwoFactorCode.js
│   ├── 📄 User.js
│   └── 📄 index.js
├── 📁 public
│   ├── 📁 css
│   ├── 📁 img
│   │   ├── 📁 articles
│   │   │   ├── 🖼️ artic1.webp
│   │   │   ├── 🖼️ artic2.jpg
│   │   │   └── 🖼️ artic3.webp
│   │   ├── 📁 new
│   │   │   ├── 🖼️ Early-Blight-Disease-Treatment-Control-2048x1152.webp
│   │   │   └── 🖼️ test.png
│   │   ├── 🖼️ appstore.png
│   │   ├── 🖼️ goolgeplay.png
│   │   ├── 🖼️ image.png
│   │   ├── 🖼️ plant-background.jpg
│   │   ├── 🖼️ plant.png
│   │   ├── 🖼️ step1.png
│   │   ├── 🖼️ step2.png
│   │   ├── 🖼️ step3.png
│   │   ├── 🖼️ step4.png
│   │   ├── 🖼️ user-profile.png
│   │   ├── 🖼️ user1.png
│   │   ├── 🖼️ user2.png
│   │   └── 🖼️ user3.png
│   ├── 📁 not-now
│   │   ├── 🌐 admin-new.html
│   │   ├── 🌐 admin.html
│   │   ├── 🌐 adminSub.html
│   │   ├── 🌐 doc.html
│   │   └── 🌐 research-papers.html
│   ├── 📁 plants
│   │   ├── 📁 Blueberry
│   │   │   └── 🖼️ Blueberryhealthy.JPG
│   │   ├── 📁 Cherry
│   │   │   ├── 🖼️ CherryPowderymildew.JPG
│   │   │   └── 🖼️ Cherryhealthy.JPG
│   │   ├── 📁 Corn
│   │   │   ├── 🖼️ CornCommonRust1.JPG
│   │   │   ├── 🖼️ Corn_(maize)Cercospora_leaf_spot Gray_leaf_spot.JPG
│   │   │   ├── 🖼️ Corn_(maize)___Northern_Leaf_Blight.JPG
│   │   │   └── 🖼️ Corn_(maize)___healthy.jpg
│   │   ├── 📁 Grape
│   │   │   ├── 🖼️ Grape___Black_rot.JPG
│   │   │   ├── 🖼️ Grape___Esca_(Black_Measles).JPG
│   │   │   ├── 🖼️ Grape___Leaf_blight_(Isariopsis_Leaf_Spot).JPG
│   │   │   └── 🖼️ Grape___healthy.JPG
│   │   ├── 📁 Orange
│   │   │   └── 🖼️ Orange___Haunglongbing_(Citrus_greening).JPG
│   │   ├── 📁 Peach
│   │   │   ├── 🖼️ Peach___Bacterial_spot.JPG
│   │   │   └── 🖼️ Peach___healthy.JPG
│   │   ├── 📁 Pepper
│   │   │   ├── 🖼️ Pepper,_bell___Bacterial_spot.JPG
│   │   │   ├── 🖼️ Pepper,_bell___healthy.JPG
│   │   │   └── 🖼️ Potato___Early_blight.JPG
│   │   ├── 📁 Potato
│   │   │   ├── 🖼️ Potato___Early_blight.JPG
│   │   │   ├── 🖼️ Potato___Late_blight.JPG
│   │   │   └── 🖼️ Potato___healthy.JPG
│   │   ├── 📁 Raspberry
│   │   │   └── 🖼️ Raspberry___healthy.JPG
│   │   ├── 📁 Soybean
│   │   │   └── 🖼️ Soybean___healthy.JPG
│   │   ├── 📁 Squash
│   │   │   └── 🖼️ Squash___Powdery_mildew.JPG
│   │   ├── 📁 Strawberry
│   │   │   ├── 🖼️ Strawberry___Leaf_scorch.JPG
│   │   │   └── 🖼️ Strawberry___healthy.JPG
│   │   ├── 📁 Tomato
│   │   │   ├── 🖼️ Tomato___Bacterial_spot.JPG
│   │   │   ├── 🖼️ Tomato___Early_blight.JPG
│   │   │   ├── 🖼️ Tomato___Late_blight.JPG
│   │   │   ├── 🖼️ Tomato___Leaf_Mold.JPG
│   │   │   ├── 🖼️ Tomato___Septoria_leaf_spot.JPG
│   │   │   ├── 🖼️ Tomato___Spider_mites Two-spotted_spider_mite.JPG
│   │   │   ├── 🖼️ Tomato___Target_Spot.JPG
│   │   │   ├── 🖼️ Tomato___Tomato_Yellow_Leaf_Curl_Virus.JPG
│   │   │   ├── 🖼️ Tomato___Tomato_mosaic_virus.JPG
│   │   │   └── 🖼️ Tomato___healthy.JPG
│   │   └── 📁 apple
│   │       ├── 🖼️ AppleBlackrot.JPG
│   │       ├── 🖼️ AppleCedarRust1.JPG
│   │       ├── 🖼️ AppleScab1.JPG
│   │       └── 🖼️ Applehealthy.JPG
│   ├── 📁 scripts
│   │   ├── 📄 admin.js
│   │   ├── 📄 contact.js
│   │   ├── 📄 dashboard.js
│   │   ├── 📄 forgot-password.js
│   │   ├── 📄 header.js
│   │   ├── 📄 history.js
│   │   ├── 📄 library.js
│   │   ├── 📄 login.js
│   │   ├── 📄 plant.js
│   │   ├── 📄 profile.js
│   │   ├── 📄 recommendations.js
│   │   ├── 📄 reminders.js
│   │   ├── 📄 reset-password.js
│   │   ├── 📄 scan.js
│   │   ├── 📄 subscribe.js
│   │   ├── 📄 treatment.js
│   │   └── 📄 weather.js
│   ├── 🌐 about-us.html
│   ├── 🌐 adding-files.html
│   ├── 🌐 adminSub.html
│   ├── 🌐 advertisement.html
│   ├── 🌐 contact.html
│   ├── 🌐 dashboard.html
│   ├── 🌐 disease-library.html
│   ├── 🌐 dmca.html
│   ├── 🌐 donate.html
│   ├── 🌐 forgot-password.html
│   ├── 🌐 help.html
│   ├── 🌐 history.html
│   ├── 🌐 home.html
│   ├── 🌐 index.html
│   ├── 🌐 indexdev.html
│   ├── 🌐 login.html
│   ├── 🌐 official-rules.html
│   ├── 🌐 plants.html
│   ├── 🌐 privacy-policy.html
│   ├── 🌐 profile.html
│   ├── 🌐 recommendation.html
│   ├── 🌐 register.html
│   ├── 🌐 reminders.html
│   ├── 🌐 reset-password.html
│   ├── 🌐 scan.html
│   ├── 🌐 subscribe.html
│   ├── 🌐 terms.html
│   ├── 🌐 treatment.html
│   └── 🌐 weather.html
├── 📁 routes
│   ├── 📄 admin.js
│   ├── 📄 adminSubscriptions.js
│   ├── 📄 analytics.js
│   ├── 📄 auth.js
│   ├── 📄 chat.js
│   ├── 📄 classify.js
│   ├── 📄 contact.js
│   ├── 📄 feedback.js
│   ├── 📄 forum.js
│   ├── 📄 general.js
│   ├── 📄 guest.js
│   ├── 📄 history.js
│   ├── 📄 index.js
│   ├── 📄 notification.js
│   ├── 📄 plants.js
│   ├── 📄 posts.js
│   ├── 📄 recommendationRoutes.js
│   ├── 📄 reminders.js
│   ├── 📄 reports.js
│   ├── 📄 subscription.js
│   ├── 📄 treatment.js
│   └── 📄 weather.js
├── 📁 test
├── 📁 uploads
│   ├── 📁 2ab9d227-2420-4f26-974e-474e252854e0
│   │   ├── 🖼️ PotatoHealthy2-1746978295916-4cee9211.jpeg
│   │   └── 🖼️ b2600118-800px-wm-1751451383175-fb364c64.jpg
│   ├── 📁 4c642cbd-b51a-4ca8-8ab5-5e0dace3cf67
│   │   └── 🖼️ AppleCedarRust1-1752247645794-24f28feb.JPG
│   ├── 📁 548d8f65-b5f7-42ad-b928-846e8d5baa93
│   │   ├── 🖼️ AppleCedarRust1-1742793756033.JPG
│   │   ├── 🖼️ AppleCedarRust1-1742794028211.JPG
│   │   └── 🖼️ AppleScab1-1742793998340.JPG
│   ├── 📁 67cf8380ee7c7f4c3915d14d
│   │   └── 🖼️ CornCommonRust1-1741655108883.JPG
│   ├── 📁 67cf862b1728ed3ffc473bfc
│   │   ├── 🖼️ 00a6039c-e425-4f7d-81b1-d6b0e668517e___RS_HL 7669-1741656219547.JPG
│   │   ├── 🖼️ ......
│   ├── 📁 67d07c15b4acd2eca111e638
│   │   ├── 🖼️ 04-1744578691532.jpg
│   │   ├── 🖼️ AppleBlackrot-1742571106731.JPG
│   │   └── 🖼️ ......
├── 📁 utils
│   ├── 📄 formatDate.js
│   ├── 📄 index.js
│   ├── 📄 mailer.js
│   ├── 📄 recommendationEngine.js
│   └── 📄 whatsappValidation.js
├── ⚙️ .gitignore
├── 📄 app.js
├── 📄 log.txt
├── ⚙️ package-lock.json
├── ⚙️ package.json
├── 📄 server.js
├── 📄 staticRoutes.js
└── 📄 test-email.js
```

### Python AI Backend

```
aibackend-pestector/
├── models/  # Trained ML models
│   ├── plant_disease_vit_BEST_model_state.pth         
│   └── vgg_model.h5
├── static/ # Simple UI
│   └── HTML,CSS,JS Files          
├── uploads/ # User Images
│   └── ...images.png          
├── requirements.txt
├── treatment_recommendations.json
├── reason.json
├── app_combined_v2_2_5.log    # Log File
└── app.py             # FastAPI app
```

---

## 📈 Model Performance

### Training Metrics

- **Training Accuracy**: ~98%
- **Validation Accuracy**: ~96%
- **Test Accuracy**: ~95%
- **Training Time**: ~2 hours on GPU
- **Model Size**: ~50 MB

### Performance Benchmarks

- **Average Prediction Time**: < 500ms
- **Image Preprocessing**: < 100ms
- **Model Inference**: < 300ms
- **Response Time (End-to-End)**: < 1s

### Confusion Matrix

The model shows high accuracy across all 38 classes with minimal misclassification between visually similar disease categories.

---

## 🤝 Contributing

We welcome contributions! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

### Development Guidelines

- Follow existing code style
- Write descriptive commit messages
- Add tests for new features
- Update documentation as needed

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- **Dataset**: [New Plant Diseases Dataset](https://www.kaggle.com/datasets/vipoooool/new-plant-diseases-dataset) by Samir Bhattarai
- **Deep Learning Framework**: TensorFlow/Keras team
- **Community**: Open-source contributors and agricultural technology enthusiasts

---

## 📞 Contact

- **Developer**: Abdelrahman968
- **GitHub**: [@Abdelrahman968](https://github.com/Abdelrahman968)
- **Email**: [Contact via GitHub](https://github.com/Abdelrahman968)

---

## 🔮 Future Enhancements

- [✅] Mobile application (iOS & Android)
- [ ] Real-time camera detection
- [✅] Treatment recommendations
- [ ] Multilingual support
- [ ] Offline mode capability
- [ ] Integration with IoT sensors
- [✅] Advanced analytics dashboard
- [✅] Community forum for farmers

---

## 📸 Screenshots

*Coming soon*

---

**Made with ❤️ for sustainable agriculture**
