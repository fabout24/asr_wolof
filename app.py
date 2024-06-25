from transformers import pipeline
import os
import librosa
import warnings
warnings.filterwarnings('ignore')


from transformers import WhisperProcessor, WhisperForConditionalGeneration
import librosa
import gradio as gr 
import warnings
warnings.filterwarnings('ignore')


MODEL_PATH =  "Whisper_ASR/"

# load model and processor
processor = WhisperProcessor.from_pretrained("Whisper_ASR/")
model = WhisperForConditionalGeneration.from_pretrained("Whisper_ASR/")


def SpeechToText(audio):

    speech_array, _= librosa.load(audio)
    print(speech_array)
    input_features = processor(speech_array,sampling_rate=16000, return_tensors="pt").input_features

    # generate token ids
    predicted_ids = model.generate(input_features)
    # decode token ids to text

    transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)
    print(transcription)
    return transcription


inputs = gr.inputs.Audio(label="Input Audio", type="filepath")
outputs = gr.outputs.Textbox(label="Output Text")
model_name = "Baamtu-Wolof-ASR"
title = model_name
description = f"Gradio demo for a {model_name}. To use it, simply upload your audio, or click one of the examples to load them. Read more at the links below. Currently supports .wav 16_000hz files"
examples = [
    ["test1.wav"],
    ["test2.wav"],
    ["test3.wav"],
    ["test4.wav"],
    ["test5.wav"],
]
print("Gradio Web UI")
iface = gr.Interface(
    SpeechToText,
    inputs,
    outputs,
    title=title,
    description=description,
    examples=examples
)
iface.launch(server_name="0.0.0.0",share=False)


