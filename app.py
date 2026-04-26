import os
import tempfile

import gradio as gr
from huggingface_hub import InferenceClient
from pydub import AudioSegment
import io

HF_TOKEN = os.environ.get("HF_TOKEN")

MODELS = {
    "Kokoro-82M ⭐ Recommended": "hexgrad/Kokoro-82M",
    "Spark-TTS-0.5B":           "SparkAudio/Spark-TTS-0.5B",
    "Orpheus-3B":                "canopylabs/orpheus-3b-0.1-ft",
    "Chatterbox":                "ResembleAI/chatterbox",
}

MODEL_LABELS = list(MODELS.keys())


def _audio_bytes_to_mp3(audio_bytes: bytes) -> str:
    segment = AudioSegment.from_file(io.BytesIO(audio_bytes))
    tmp = tempfile.NamedTemporaryFile(suffix=".mp3", delete=False)
    segment.export(tmp.name, format="mp3")
    return tmp.name


def generate(text: str, model_label: str) -> tuple[str, str]:
    if not text.strip():
        raise gr.Error("Please enter some text first.")

    model_id = MODELS[model_label]
    client = InferenceClient(token=HF_TOKEN)

    try:
        audio_bytes = client.text_to_speech(text, model=model_id)
        mp3_path = _audio_bytes_to_mp3(audio_bytes)
        return mp3_path, "Done!"
    except Exception as exc:
        raise gr.Error(f"Generation failed: {exc}") from exc


with gr.Blocks(title="StoryToSleep TTS", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🎙️ StoryToSleep TTS")
    gr.Markdown("Paste your text, choose a model, and download the MP3.")

    with gr.Row():
        with gr.Column(scale=2):
            text_input = gr.Textbox(
                label="Text",
                placeholder="Enter the text you'd like read aloud...",
                lines=10,
            )
            with gr.Row():
                model_dropdown = gr.Dropdown(
                    choices=MODEL_LABELS,
                    value=MODEL_LABELS[0],
                    label="Model",
                    scale=3,
                )
                generate_btn = gr.Button("Generate", variant="primary", scale=1)

        with gr.Column(scale=1):
            audio_output = gr.Audio(
                label="Output (playback + download)",
                type="filepath",
                format="mp3",
            )
            status = gr.Textbox(label="Status", interactive=False, max_lines=1)

    generate_btn.click(
        fn=generate,
        inputs=[text_input, model_dropdown],
        outputs=[audio_output, status],
    )

if __name__ == "__main__":
    demo.launch(share=True)
