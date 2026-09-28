import gradio as gr
import torch
import librosa
from model import AudioSpoofDetector, preprocess_audio

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load model weights
model = AudioSpoofDetector().to(DEVICE)
try:
    model.load_state_dict(torch.load("spoof_detector.pth", map_location=DEVICE))
    model.eval()
    print("Model loaded successfully from spoof_detector.pth")
except FileNotFoundError:
    print("Warning: 'spoof_detector.pth' not found. Please run train.py first.")

def predict_audio(audio_file):
    if audio_file is None:
        return None, "<div class='result-card alert'>⚠️ Please record or upload an audio file first.</div>"

    try:
        # Load audio via Librosa at 16kHz mono
        audio_np, sr = librosa.load(audio_file, sr=16000, mono=True)
        if len(audio_np) == 0:
            return None, "<div class='result-card alert'>⚠️ Audio recording was empty. Please try again.</div>"

        # Extract features
        features = preprocess_audio(audio_np, sr)
        if features.dim() == 3:
            features = features.unsqueeze(0)  # Shape: (1, 3, N_MFCC, Time)
            
        features = features.to(DEVICE)

        # Inference
        with torch.no_grad():
            logit = model(features).squeeze()
            prob_spoof = torch.sigmoid(logit / 1.1).item()
            prob_real = 1.0 - prob_spoof

        confidence_scores = {
            "Real Voice": float(prob_real),
            "Spoof / Fake Voice": float(prob_spoof)
        }
        
        # --- SYNCHRONIZED DECISION LOGIC ---
        # Binary decision set at standard 50% boundary to match gr.Label top prediction
        if prob_spoof >= 0.50:
            verdict_title = "SPOOF / FAKE VOICE DETECTED"
            confidence_pct = prob_spoof * 100
            card_class = "spoof-card"
            badge_icon = "🚨"
        else:
            verdict_title = "REAL VOICE DETECTED"
            confidence_pct = prob_real * 100
            card_class = "real-card"
            badge_icon = "✅"

        summary_html = f"""
        <div class="result-card {card_class}">
            <div class="card-header">
                <span class="badge-icon">{badge_icon}</span>
                <span class="verdict-text">{verdict_title}</span>
            </div>
            <div class="card-body">
                <div class="confidence-label">Confidence Score</div>
                <div class="confidence-value">{confidence_pct:.1f}%</div>
            </div>
        </div>
        """

        return confidence_scores, summary_html

    except Exception as e:
        return None, f"<div class='result-card alert'>❌ Error processing audio: {str(e)}</div>"

custom_css = """
.gradio-container {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
    max-width: 900px !important;
    margin: 30px auto !important;
    padding: 20px !important;
    border-radius: 16px !important;
    box-shadow: 0 10px 30px rgba(0,0,0,0.08) !important;
}

.main-title {
    text-align: center;
    font-size: 2rem;
    font-weight: 700;
    margin-bottom: 6px;
    color: #1e293b;
}

.sub-title {
    text-align: center;
    font-size: 0.98rem;
    color: #64748b;
    margin-bottom: 24px;
}

.fixed-col {
    min-height: 380px !important;
    max-height: 380px !important;
    display: flex !important;
    flex-direction: column !important;
    justify-content: space-between !important;
}

.result-container {
    min-height: 120px !important;
    max-height: 120px !important;
    overflow: hidden !important;
}

.result-card {
    padding: 16px 20px;
    border-radius: 12px;
    border: 1px solid transparent;
    transition: all 0.3s ease;
}

.real-card {
    background-color: #f0fdf4;
    border-color: #bbf7d0;
    color: #166534;
}

.spoof-card {
    background-color: #fef2f2;
    border-color: #fecaca;
    color: #991b1b;
}

.alert {
    background-color: #fffbe3;
    border-color: #ffe58f;
    color: #856404;
}

.card-header {
    display: flex;
    align-items: center;
    gap: 10px;
    margin-bottom: 8px;
}

.badge-icon {
    font-size: 1.3rem;
}

.verdict-text {
    font-weight: 700;
    font-size: 1.1rem;
    letter-spacing: 0.5px;
}

.card-body {
    display: flex;
    align-items: baseline;
    gap: 12px;
}

.confidence-label {
    font-size: 0.85rem;
    opacity: 0.8;
    text-transform: uppercase;
    font-weight: 600;
}

.confidence-value {
    font-size: 1.4rem;
    font-weight: 800;
}

.analyze-btn {
    background: linear-gradient(135deg, #4f46e5 0%, #3b82f6 100%) !important;
    color: white !important;
    font-weight: 600 !important;
    border-radius: 8px !important;
    border: none !important;
    box-shadow: 0 4px 12px rgba(79, 70, 229, 0.25) !important;
    transition: transform 0.1s ease, box-shadow 0.2s ease !important;
}

.analyze-btn:hover {
    box-shadow: 0 6px 16px rgba(79, 70, 229, 0.35) !important;
}
"""

with gr.Blocks(css=custom_css, title="Audio Authenticity Inspector") as demo:
    gr.HTML("""
        <div class="main-title">🎙️ Deepfake Voice Authenticity Inspector</div>
        <div class="sub-title">Analyze audio samples using multi-channel MFCC feature modeling to detect synthetic speech.</div>
    """)

    with gr.Row():
        with gr.Column(elem_classes=["fixed-col"]):
            audio_input = gr.Audio(
                sources=["microphone", "upload"],
                type="filepath",
                label="Input Audio Sample"
            )
            analyze_btn = gr.Button("Analyze Audio", variant="primary", elem_classes=["analyze-btn"])

        with gr.Column(elem_classes=["fixed-col"]):
            output_label = gr.Label(num_top_classes=2, label="Prediction Probability")
            summary_box = gr.HTML(
                value="""
                <div class="result-card" style="background-color: #f8fafc; border-color: #e2e8f0; color: #64748b;">
                    <div class="card-header">
                        <span class="badge-icon">ℹ️</span>
                        <span class="verdict-text">Ready for Analysis</span>
                    </div>
                    <div class="card-body">
                        <span style="font-size: 0.9rem;">Record or upload an audio clip to view the detection result.</span>
                    </div>
                </div>
                """,
                elem_classes=["result-container"]
            )

    analyze_btn.click(
        fn=predict_audio,
        inputs=[audio_input],
        outputs=[output_label, summary_box]
    )

if __name__ == "__main__":
    # Setting inbrowser=False stops Chrome from opening automatically
    demo.launch(server_name="127.0.0.1", server_port=7860, inbrowser=False, theme=gr.themes.Soft())