# AI-Powered Sustainable Agriculture Advisor

A comprehensive crop disease prediction and recommendation system that uses deep learning and RAG (Retrieval-Augmented Generation) to help farmers identify plant diseases and get treatment recommendations.

## 🌟 Features

- **Image-based Disease Detection**: CNN model trained on 87K+ images across 38 crop disease classes
- **AI-Powered Recommendations**: RAG pipeline using Groq LLM for personalized treatment advice
- **Comprehensive Knowledge Base**: Covers major crops including Apple, Tomato, Corn, Potato, Grape, and more
- **Real-time API**: FastAPI backend with automatic database storage and retrieval
- **User-friendly Interface**: Streamlit frontend for easy image upload and result viewing
- **Vector Database**: FAISS-based semantic search for relevant disease information
- **Production Ready**: MySQL database, Docker support, and comprehensive testing

## 🏗️ Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Frontend      │    │   FastAPI       │    │   ML Pipeline   │
│   (Streamlit)   │◄──►│   Backend       │◄──►│   (TensorFlow)  │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                                │
                                ▼
                       ┌─────────────────┐
                       │   RAG Pipeline  │
                       │  (FAISS + Groq) │
                       └─────────────────┘
                                │
                                ▼
                       ┌─────────────────┐
                       │   MySQL DB      │
                       │  (Predictions)  │
                       └─────────────────┘
```

## 🚀 Quick Start

### 1. Prerequisites

- Python 3.8+
- MySQL Server 8.0+
- CUDA-compatible GPU (optional, for faster inference)
- Groq API key for RAG functionality

### 2. Installation

```bash
# Clone the repository
git clone <repository-url>
cd crop_advisor

# Install dependencies
pip install -r requirements.txt

# Run setup script
python scripts/setup.py
```

### 3. Database Setup

```bash
# Set up MySQL database
python setup_mysql.py

# Or manually:
# 1. Create database: CREATE DATABASE crop_advisor;
# 2. Run migrations: mysql -u root -p crop_advisor < database/migrations/init_db.sql
```

### 4. Configuration

1. **Update `.env` file** with your credentials:
```bash
GROQ_API_KEY=your_groq_api_key_here
DB_HOST=localhost
DB_PORT=3306
DB_NAME=crop_advisor
DB_USER=root
DB_PASSWORD=your_mysql_password_here
```

2. **Download the Plant Disease Dataset**:
   - Visit: [Kaggle Plant Disease Dataset](https://www.kaggle.com/datasets/abdallahalidev/plantvillage)
   - Extract to: `data/raw/New Plant Diseases Dataset(Augmented)/`

### 5. Start the Services

```bash
# Option 1: Use startup script (recommended)
python start.py

# Option 2: Start manually
# Terminal 1: python api/main.py
# Terminal 2: streamlit run frontend/app.py
```

## 📁 Project Structure

```
crop_advisor/
├── api/                    # FastAPI backend
│   ├── main.py           # Main application
│   ├── routes/           # API endpoints
│   └── utils/            # Utility functions
├── frontend/              # Streamlit frontend
│   ├── app.py            # Main app
│   └── components/       # UI components
├── models/                # ML models and training
│   ├── classification/   # CNN training scripts
│   └── saved_models/     # Trained models
├── rag/                   # RAG pipeline
│   ├── embedder.py       # Vector database creation
│   ├── retriever.py      # Information retrieval
│   └── generator.py      # AI recommendation generation
├── database/              # Database utilities
│   ├── db_utils.py       # SQLAlchemy helpers
│   └── migrations/       # MySQL schema
├── data/                  # Data storage
│   ├── raw/              # Raw dataset
│   ├── processed/        # Processed data
│   └── embeddings/       # FAISS vector database
├── scripts/               # Utility scripts
│   └── setup.py          # Setup script
├── setup_mysql.py         # MySQL setup helper
├── start.py               # System startup script
└── test_system.py         # System testing
```

## 🔧 API Endpoints

### Prediction
- `POST /api/v1/predict` - Upload image for disease prediction
- `GET /api/v1/model-status` - Check model availability

### Data Management
- `GET /api/v1/predictions` - Get prediction history
- `GET /api/v1/predictions/{id}` - Get specific prediction
- `DELETE /api/v1/predictions/{id}` - Delete prediction
- `GET /api/v1/statistics` - Get database statistics
- `GET /api/v1/classes` - Get all disease classes
- `GET /api/v1/health` - Health check

### Recommendations
- `GET /api/v1/recommendations/{disease}` - Get treatment recommendations
- `GET /api/v1/recommendations` - Search for recommendations
- `GET /api/v1/prevention-tips` - Get prevention tips
- `GET /api/v1/disease-info/{disease}` - Get detailed disease information

## 🧠 Model Details

### CNN Architecture
- **Input**: 224x224 RGB images
- **Architecture**: Conv2D + MaxPooling2D + Dense layers
- **Classes**: 38 crop disease categories
- **Accuracy**: ~85% on validation set
- **Training**: Data augmentation (rotation, flip, zoom)

### RAG Pipeline
- **Embeddings**: Sentence Transformers (all-MiniLM-L6-v2)
- **Vector DB**: FAISS for similarity search
- **LLM**: Groq (llama-3.3-70b-versatile)
- **Knowledge Base**: Comprehensive disease treatment database

## 📊 Supported Crops and Diseases

### Apple
- Apple scab, Cedar apple rust, Black rot, Healthy

### Tomato
- Bacterial spot, Early blight, Late blight, Leaf mold, Septoria leaf spot, Healthy

### Corn
- Gray leaf spot, Common rust, Healthy

### Potato
- Early blight, Late blight, Healthy

### Grape
- Black rot, Esca, Healthy

### Other Crops
- Cherry, Peach, Bell Pepper, Strawberry, Soybean, Raspberry, Squash, Orange

## 🐳 Docker Deployment

```bash
# Build and run with Docker Compose
docker-compose up --build

# Or build individual containers
docker build -f docker/Dockerfile.api -t crop-advisor-api .
docker build -f docker/Dockerfile.frontend -t crop-advisor-frontend .
```

## 🧪 Testing

```bash
# Run comprehensive system tests
python test_system.py

# Test individual components
python -m pytest tests/
```

## 📈 Performance

- **Prediction Speed**: ~200ms per image (CPU), ~50ms (GPU)
- **API Response Time**: <500ms for full prediction + recommendation
- **Model Size**: ~50MB (compressed)
- **Database**: MySQL with optimized indexes
- **Vector Search**: FAISS with sub-second response times

## 🔒 Security Features

- Input validation and sanitization
- File type restrictions
- Rate limiting (configurable)
- CORS configuration
- Error handling without information leakage

## 🚀 Production Deployment

### AWS RDS + EC2
- Managed MySQL database with automatic backups
- Auto-scaling EC2 instances
- Load balancer for high availability

### GCP Cloud SQL + Cloud Run
- Managed MySQL with automatic failover
- Serverless container deployment
- Global load balancing

### Azure Database + Container Instances
- Managed MySQL with geo-replication
- Container-based deployment
- Traffic manager for global routing

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Submit a pull request

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- PlantVillage dataset contributors
- TensorFlow and FastAPI communities
- Groq for LLM API access
- FAISS for vector similarity search

## 📞 Support

- **Issues**: GitHub Issues
- **Documentation**: `/docs` endpoint when API is running
- **Email**: [your-email@example.com]

## 🔄 Version History

- **v1.0.0**: Initial release with CNN + RAG pipeline
- **v1.1.0**: Added comprehensive API endpoints
- **v1.2.0**: Enhanced frontend and database utilities
- **v1.3.0**: Switched to MySQL database
- **v1.4.0**: Simplified project structure

---

**Built with ❤️ for sustainable agriculture**