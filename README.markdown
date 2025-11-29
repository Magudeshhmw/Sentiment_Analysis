# 🌟 Advanced Sentiment Analysis AI 🌟

![Sentiment Analysis Banner](https://img.shields.io/badge/Sentiment-Analysis-%2300C4B4?style=for-the-badge&logo=python&logoColor=white)
![Python](https://img.shields.io/badge/Python-3.8%2B-%2314354C?style=flat-square&logo=python)
![Flask](https://img.shields.io/badge/Flask-2.3-%23000000?style=flat-square&logo=flask)
![Scikit-Learn](https://img.shields.io/badge/scikit--learn-%23F7931E?style=flat-square&logo=scikit-learn)
![License](https://img.shields.io/badge/License-MIT-%23FF5555?style=flat-square)

---

## 📋 Project Overview

Welcome to the **Advanced Sentiment Analysis AI**! 🚀 This project leverages a hybrid approach combining **Machine Learning (LinearSVC)** and **Lexicon-based analysis (VADER)** to provide highly accurate sentiment detection.

The application features a stunning, modern web interface with real-time visualizations, making it not just a tool, but an experience. ✨

![Dashboard Preview](assets/dashboard_preview.png)

---

## 🚀 Key Features

- **Hybrid Intelligence**: Combines the statistical power of LinearSVC (trained on movie reviews) with the rule-based precision of VADER. 🧠
- **Real-time Visualization**: Interactive doughnut charts display probability distributions (Positive vs Negative). 📊
- **Modern UI/UX**: A glassmorphism-inspired dark theme with smooth animations and responsive design. 🎨
- **Detailed Metrics**: View model confidence, probability scores, and VADER compound scores. 📈
- **Robust API**: A RESTful Flask API that powers the frontend and can be integrated into other systems. 🔌

---

## 🏗️ Technical Architecture

The system follows a modular client-server architecture.

```mermaid
graph TD
    subgraph Client [Frontend (Browser)]
        UI[User Interface (HTML/CSS/JS)]
        Input[Text Input]
        Chart[Chart.js Visualization]
    end

    subgraph Server [Backend (Flask)]
        API[Flask API (/analyze)]
        
        subgraph NLP_Pipeline [NLP Pipeline]
            Pre[Preprocessing (NLTK)]
            Vector[TF-IDF Vectorizer]
        end
        
        subgraph Models [Inference Engine]
            SVC[LinearSVC Model (Scikit-learn)]
            VADER[VADER Analyzer (NLTK)]
            Ensemble[Logic Layer (Hybrid Decision)]
        end
    end

    %% Data Flow
    Input -->|1. User enters text| UI
    UI -->|2. POST Request (JSON)| API
    API -->|3. Raw Text| Pre
    Pre -->|4. Cleaned Tokens| Vector
    Vector -->|5. Feature Vector| SVC
    API -->|3b. Raw Text| VADER
    
    SVC -->|6. Probability & Prediction| Ensemble
    VADER -->|6b. Compound Score| Ensemble
    
    Ensemble -->|7. Final Sentiment & Metrics| API
    API -->|8. JSON Response| UI
    UI -->|9. Render Results| Chart
```

---

## 🛠️ Installation & Setup

Follow these steps to get the project running on your local machine.

### Prerequisites
- **Python 3.8+** 🐍
- **pip** (Python package manager)

### Steps

1.  **Clone the Repository**:
    ```bash
    git clone <your-repo-url>
    cd sentiment-analysis-app
    ```

2.  **Install Dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

3.  **Run the Application**:
    ```bash
    python app.py
    ```

4.  **Access the App**:
    Open your browser and navigate to:
    `http://localhost:5000`

---

## 📖 Usage Guide

1.  **Enter Text**: Type or paste any text into the input area.
    *   *Example: "The cinematography was breathtaking, but the plot felt a bit rushed."*
2.  **Analyze**: Click the **Analyze Text** button.
3.  **View Results**:
    *   **Sentiment Badge**: See the overall sentiment (Positive, Negative, or Neutral).
    *   **Confidence**: Check how confident the model is in its prediction.
    *   **Probability Chart**: Visualize the balance between positive and negative signals.

---

## 📂 Project Structure

```
sentiment-analysis-app/
├── app.py                  # Main Flask application & ML logic
├── requirements.txt        # Python dependencies
├── templates/
│   └── index.html          # Modern Frontend (HTML/CSS/JS)
├── sentiment_analysis.py   # Console version (Legacy)
└── README.markdown         # Project Documentation
```

---

## 🤝 Contributing

Contributions are welcome! 🙌
1.  Fork the repository.
2.  Create a feature branch (`git checkout -b feature/AmazingFeature`).
3.  Commit your changes (`git commit -m 'Add some AmazingFeature'`).
4.  Push to the branch (`git push origin feature/AmazingFeature`).
5.  Open a Pull Request.

---

## 📜 License

This project is licensed under the MIT License.

---

⭐ **Star this project on GitHub if you find it useful!** ⭐