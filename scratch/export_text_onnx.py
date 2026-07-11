import os
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForSequenceClassification

def main():
    print("Loading Hugging Face model 'j-hartmann/emotion-english-distilroberta-base'...")
    model_name = "j-hartmann/emotion-english-distilroberta-base"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    raw_model = AutoModelForSequenceClassification.from_pretrained(model_name)
    
    # Define wrapper to output both logits (for text prediction) and CLS embedding (for early fusion)
    class TextPipelineWrapper(nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model
        def forward(self, input_ids, attention_mask):
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
            logits = outputs.logits
            # The last hidden state CLS token (index 0)
            cls_embedding = outputs.hidden_states[-1][:, 0, :]
            return logits, cls_embedding

    wrapped_model = TextPipelineWrapper(raw_model)
    wrapped_model.eval()
    
    # Create dummy inputs for tokenized text (batch_size=1, seq_len=16)
    dummy_text = "I am feeling extremely happy today!"
    inputs = tokenizer(dummy_text, return_tensors="pt")
    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]
    
    print("Exporting Text pipeline to ONNX...")
    os.makedirs("onnx_models", exist_ok=True)
    
    torch.onnx.export(
        wrapped_model,
        (input_ids, attention_mask),
        "onnx_models/text_pipeline.onnx",
        input_names=["input_ids", "attention_mask"],
        output_names=["logits", "cls_embedding"],
        dynamic_axes={
            "input_ids": {0: "batch_size", 1: "sequence_length"},
            "attention_mask": {0: "batch_size", 1: "sequence_length"},
            "logits": {0: "batch_size"},
            "cls_embedding": {0: "batch_size"}
        },
        opset_version=12
    )
    print("Text pipeline exported successfully to 'onnx_models/text_pipeline.onnx'!")

if __name__ == "__main__":
    main()
