# Lightweight Swahili ASR for On-Device Transcription

This project is a solution for the AI for Good Swahili ASR Challenge, focusing on building an efficient, offline, and privacy-preserving speech recognition system for over 200 million Swahili speakers. Our model is designed to run on edge devices with limited computational resources, making modern AI accessible to communities often left behind.

---

## 🚀 Our Solution

Our approach combines a highly optimized model with cutting-edge training techniques to achieve state-of-the-art performance within strict resource constraints.

* **Base Model**: We fine-tuned the `Abdoul27/whisper-turbo-v3-model`, a distilled version of Whisper optimized for speed. This model was already pre-trained on the Common Voice 12 Swahili dataset, giving it a strong initial understanding of the language.
* **Efficient Fine-Tuning**: To train on a single NVIDIA T4 GPU (≤16 GB VRAM), we used:
    * **PEFT/LoRA**: Froze the base model and only trained tiny, efficient "adapter" layers.
    * **8-bit Quantization**: Loaded the model in a lower-precision format to drastically reduce the memory footprint.
* **Robust Training Data**: We trained the model on real-world, conversational Swahili data (`Sunbird/salt`) and used noise augmentation with urban sounds from East Africa to ensure it performs well in noisy, everyday environments.
* **Accurate Inference**: We used beam search decoding during transcription to improve accuracy and reduce errors.

---

## ⚙️ Hardware, Libraries & Training Parameters

**Hardware Used**: NVIDIA T4 GPU (16 GB VRAM) on Kaggle Notebooks.

**Key Libraries**:
* `transformers`
* `peft` (for LoRA)
* `datasets`
* `torch`
* `bitsandbytes` (for 8-bit quantization)
* `evaluate` (for WER/CER metrics)

**Key Training Parameters**:
* **Effective Batch Size**: 8 (`per_device_train_batch_size` of 4 with 2 `gradient_accumulation_steps`)
* **Learning Rate**: 1e-5
* **Warmup Steps**: 500
* **Epochs**: 3
* **Precision**: Mixed Precision (`fp16`) enabled.
* **Memory Optimization**: `gradient_checkpointing` enabled.

---

## 📊 Performance

Our solution delivers a powerful combination of accuracy and speed, meeting all challenge requirements.

* **Final Private Leaderboard WER**: 17.81%
* **Inference Speed**: ~1.24 seconds per audio file on average.
* **Resource Efficiency**: Runs comfortably within the 16 GB VRAM limit of an NVIDIA T4 GPU. 

---

## 🛠️ Reproducibility

The repository includes the necessary scripts to reproduce our results:
* A training script to fine-tune the model.
* An inference script to generate transcriptions.
* The final trained model artifacts.

This project demonstrates that it is possible to build high-performance, privacy-focused AI solutions for low-resource languages that can run effectively on accessible hardware.
