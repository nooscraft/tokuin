#!/usr/bin/env python3
"""
Alternative script to convert sentence-transformers model to ONNX format.
Use this if optimum-cli doesn't work.

This script will automatically use .venv/bin/python3 if available,
or you can activate your venv first:
  source .venv/bin/activate
  python3 scripts/convert-model.py
"""

import sys
import os
from pathlib import Path

# Try to use .venv if it exists
script_dir = Path(__file__).parent.parent
venv_python = script_dir / ".venv" / "bin" / "python3"
if venv_python.exists():
    # If we're not already using the venv python, suggest using it
    current_python = sys.executable
    if str(venv_python) not in current_python:
        print(f"💡 Note: Found .venv at {script_dir / '.venv'}")
        print(f"   Current Python: {current_python}")
        print(f"   To use venv, run: source .venv/bin/activate && python3 {__file__}")
        print("   Or the script will work with your current Python environment\n")

# Try different import methods
try:
    # Method 1: Try optimum.onnxruntime (optimum < 2.0)
    from optimum.onnxruntime import ORTModelForFeatureExtraction
    from transformers import AutoTokenizer
    USE_ORT_MODEL = True
    USE_TRANSFORMERS_EXPORT = False
except ImportError:
    try:
        # Method 2: Use sentence-transformers (works with optimum 2.0+)
        try:
            from sentence_transformers import SentenceTransformer
            USE_SENTENCE_TRANSFORMERS = True
        except ImportError:
            from transformers import AutoTokenizer, AutoModel
            USE_SENTENCE_TRANSFORMERS = False
        import torch
        USE_ORT_MODEL = False
        USE_TRANSFORMERS_EXPORT = True
        print("💡 Using transformers + torch.onnx export method")
    except ImportError as e:
        print(f"❌ Error: Required packages not installed: {e}")
        print("\nPlease install:")
        print('  pip install transformers torch onnx onnxruntime sentence-transformers onnxscript')
        print("\nOr if you're in a venv:")
        print('  source .venv/bin/activate')
        print('  pip install transformers torch onnx onnxruntime')
        sys.exit(1)

def convert_model():
    model_id = "sentence-transformers/all-MiniLM-L6-v2"
    script_dir = Path(__file__).parent.parent
    output_dir = script_dir / "onnx_model"
    cache_dir = Path.home() / "Library/Caches/tokuin/models/all-MiniLM-L6-v2"
    
    print(f"🔄 Converting {model_id} to ONNX format...")
    print(f"   Output directory: {output_dir}")
    print("")
    
    # Create output directory
    output_dir.mkdir(exist_ok=True)
    
    try:
        if USE_ORT_MODEL:
            # Method 1: Using ORTModelForFeatureExtraction (optimum < 2.0)
            print("📥 Loading model from HuggingFace...")
            model = ORTModelForFeatureExtraction.from_pretrained(
                model_id,
                export=True,
                cache_dir=str(output_dir)
            )
            
            # Save the model
            print("💾 Saving ONNX model...")
            model.save_pretrained(str(output_dir))
            
            # Also save tokenizer
            print("💾 Saving tokenizer...")
            tokenizer = AutoTokenizer.from_pretrained(model_id)
            tokenizer.save_pretrained(str(output_dir))
        elif USE_TRANSFORMERS_EXPORT:
            # Method 2: Using sentence-transformers or transformers + torch.onnx
            if USE_SENTENCE_TRANSFORMERS:
                print("📥 Loading sentence-transformers model...")
                model = SentenceTransformer(model_id)
                # Move to CPU for export
                model = model.to("cpu")
                print("🔄 Exporting to ONNX format...")
                # Get the underlying model
                base_model = model[0].auto_model
                tokenizer = model.tokenizer
            else:
                print("📥 Loading model from HuggingFace...")
                tokenizer = AutoTokenizer.from_pretrained(model_id)
                base_model = AutoModel.from_pretrained(model_id)
            
            # Move model to CPU (required for ONNX export, MPS not supported)
            print("🔄 Moving model to CPU for ONNX export...")
            base_model = base_model.to("cpu")
            base_model.eval()  # Set to evaluation mode
            
            # Create dummy input on CPU
            print("🔄 Preparing model for export...")
            if not USE_SENTENCE_TRANSFORMERS:
                dummy_input = tokenizer("dummy text", return_tensors="pt", padding=True, truncation=True)
            else:
                dummy_input = tokenizer("dummy text", return_tensors="pt", padding=True, truncation=True)
            
            # Ensure inputs are on CPU
            dummy_input_ids = dummy_input["input_ids"].to("cpu")
            dummy_attention_mask = dummy_input.get("attention_mask", torch.ones_like(dummy_input_ids)).to("cpu")
            
            # Export to ONNX
            print("🔄 Exporting to ONNX format...")
            onnx_path = output_dir / "model.onnx"
            
            # Try the new dynamo export if onnxscript is available, otherwise use legacy
            try:
                import onnxscript
                use_dynamo = True
                print("   Using new dynamo export method...")
            except ImportError:
                use_dynamo = False
                print("   Using legacy export method (onnxscript not installed)...")
            
            try:
                if use_dynamo:
                    torch.onnx.export(
                        base_model,
                        (dummy_input_ids, dummy_attention_mask),
                        str(onnx_path),
                        input_names=["input_ids", "attention_mask"],
                        output_names=["last_hidden_state"],
                        dynamic_axes={
                            "input_ids": {0: "batch_size", 1: "sequence_length"},
                            "attention_mask": {0: "batch_size", 1: "sequence_length"},
                            "last_hidden_state": {0: "batch_size", 1: "sequence_length"}
                        },
                        opset_version=14,
                        dynamo=True,  # Use new export method
                    )
                else:
                    # Use legacy export method
                    torch.onnx.export(
                        base_model,
                        (dummy_input_ids, dummy_attention_mask),
                        str(onnx_path),
                        input_names=["input_ids", "attention_mask"],
                        output_names=["last_hidden_state"],
                        dynamic_axes={
                            "input_ids": {0: "batch_size", 1: "sequence_length"},
                            "attention_mask": {0: "batch_size", 1: "sequence_length"},
                            "last_hidden_state": {0: "batch_size", 1: "sequence_length"}
                        },
                        opset_version=14,
                    )
            except Exception as e:
                if use_dynamo:
                    # If dynamo fails, try legacy method
                    print("   ⚠️  Dynamo export failed, falling back to legacy method...")
                    torch.onnx.export(
                        base_model,
                        (dummy_input_ids, dummy_attention_mask),
                        str(onnx_path),
                        input_names=["input_ids", "attention_mask"],
                        output_names=["last_hidden_state"],
                        dynamic_axes={
                            "input_ids": {0: "batch_size", 1: "sequence_length"},
                            "attention_mask": {0: "batch_size", 1: "sequence_length"},
                            "last_hidden_state": {0: "batch_size", 1: "sequence_length"}
                        },
                        opset_version=14,
                    )
                else:
                    raise
            
            # Save tokenizer
            print("💾 Saving tokenizer...")
            if not USE_SENTENCE_TRANSFORMERS:
                tokenizer.save_pretrained(str(output_dir))
            else:
                # Tokenizer is already saved with sentence-transformers
                pass
        
        print(f"\n✅ Model converted successfully!")
        print(f"   ONNX model: {output_dir / 'model.onnx'}")
        
        # Copy to cache directory
        print(f"\n📁 Copying to cache directory...")
        cache_dir.mkdir(parents=True, exist_ok=True)
        
        model_file = output_dir / "model.onnx"
        if model_file.exists():
            import shutil
            shutil.copy2(model_file, cache_dir / "model.onnx")
            print(f"✅ Copied to: {cache_dir / 'model.onnx'}")
            
            # Get file size
            size = model_file.stat().st_size / (1024 * 1024)  # MB
            print(f"   File size: {size:.1f} MB")
        else:
            # Try to find the model file with a different name
            onnx_files = list(output_dir.glob("*.onnx"))
            if onnx_files:
                import shutil
                shutil.copy2(onnx_files[0], cache_dir / "model.onnx")
                print(f"✅ Copied to: {cache_dir / 'model.onnx'}")
            else:
                print("⚠️  Warning: Could not find model.onnx file")
                print(f"   Files in {output_dir}:")
                for f in output_dir.iterdir():
                    print(f"     - {f.name}")
        
        print("\n✅ Setup complete!")
        print("\nYou can now test with:")
        print("  ./target/release/tokuin compress your-prompt.txt --scoring semantic --model gpt-4")
        
    except Exception as e:
        print(f"\n❌ Error during conversion: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    convert_model()

