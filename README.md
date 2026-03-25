# 🏦 Banking Complaint Classifier

A Streamlit-based web application that classifies customer banking complaints into their appropriate handling categories using a fine-tuned Hugging Face transformer model.

## Features

- **AI-Powered Classification**: Uses a fine-tuned RoBERTa model for accurate complaint categorization
- **Interactive Web Interface**: Built with Streamlit for easy interaction and visualization
- **Real-time Predictions**: Instant classification with confidence scores
- **Probability Distribution**: Visualize top-K predicted categories with an interactive Plotly bar chart
- **Example Complaints**: Pre-loaded examples for quick testing
- **Detailed Analytics**: View all class probabilities in a detailed matrix

## Requirements

- Python 3.8+
- PyTorch >= 2.0.0
- Streamlit >= 1.30.0
- Transformers >= 4.30.0
- Scikit-learn >= 1.2.0
- Pandas >= 2.0.0
- Plotly >= 5.14.0
- Joblib >= 1.2.0

## Installation

1. **Clone or download the repository**:
   ```bash
   cd banking-complaint-classifier-streamlit
   ```

2. **Create a virtual environment** (optional but recommended):
   ```bash
   python -m venv venv
   
   # On Windows
   venv\Scripts\activate
   
   # On macOS/Linux
   source venv/bin/activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Ensure model files exist**:
   - `label_encoder.pkl` should be in the root directory (this file encodes the complaint categories)
   - The Hugging Face model (`Priyanshub009/banking_complaint_classfier_new`) will be auto-downloaded on first run

## Usage

1. **Run the Streamlit app**:
   ```bash
   streamlit run app.py
   ```

2. **Open in browser**:
   - The app will automatically open at `http://localhost:8501`
   - If not, navigate to the URL shown in the terminal

3. **Classify a complaint**:
   - Select a pre-loaded example from the dropdown, **or**
   - Type your own complaint in the text area
   - Click **🚀 Classify Complaint** button
   - View the predicted category and confidence score
   - Explore the probability distribution chart
   - Optionally view detailed probabilities for all categories

## Project Structure

```
banking-complaint-classifier-streamlit/
├── app.py                    # Main Streamlit application
├── label_encoder.pkl         # Encoded complaint category labels
├── requirements.txt          # Python dependencies
└── README.md                 # This file
```

## Model Information

- **Model**: `Priyanshub009/banking_complaint_classfier_new`
- **Architecture**: Transformer-based Sequence Classification
- **Framework**: Hugging Face Transformers
- **Backend**: PyTorch
- **Input**: Customer complaint text
- **Output**: Category label with confidence probability

## How It Works

1. **Tokenization**: Input complaint text is tokenized using the model's tokenizer
2. **Inference**: Tokens are passed through the transformer model to get logits
3. **Probability Conversion**: Logits are converted to probabilities using softmax
4. **Label Mapping**: Probabilities are mapped to complaint categories using the label encoder
5. **Visualization**: Top-K predictions and full probability matrix are displayed

## Example Complaints

The app includes several pre-loaded examples:
- "I was charged twice on my credit card."
- "My bank account transfer is pending for 5 days."
- "Debt collectors keep calling me for a loan I never took."
- "My loan application was rejected without explanation."
- "Money transfer failed but amount was deducted."

## Customization

### Adjust Visualization Settings
- Use the **Show Top-K Classes in Chart** slider to display more or fewer top predictions
- Toggle **Show Detailed Probabilities Table** to view all category scores

### Add More Examples
Edit the `EXAMPLES` list in `app.py` to include additional sample complaints.

### Change the Model
Replace `MODEL_DIR` value in `app.py` with a different Hugging Face model ID.

## Troubleshooting

### "Could not find label encoder file" Error
- Ensure `label_encoder.pkl` exists in the same directory as `app.py`

### Model Download Issues
- Check your internet connection (model downloads on first run)
- Verify the Hugging Face model exists and is publicly accessible
- Set HF_TOKEN environment variable if using a private model

### GPU/Memory Issues
- If running on CPU only, the model will work but be slower
- Reduce batch size if experiencing memory errors

## Performance

- **First run**: ~2-3 minutes (downloading the transformer model)
- **Subsequent runs**: <1 second per prediction (cached with `@st.cache_resource`)
- **Inference**: Real-time on CPU or GPU

## License

This project uses publicly available models and libraries. Ensure compliance with their respective licenses.

## Support & Contributions

For issues or improvements, please ensure:
1. All dependencies are installed: `pip install -r requirements.txt`
2. The `label_encoder.pkl` file is present
3. You have internet access for initial model download
