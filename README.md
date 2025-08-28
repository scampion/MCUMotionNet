# MCUMotionNet: Real-Time Object Tracking on Memory-Constrained Microcontrollers

MCUMotionNet is a lightweight, two-stage deep learning architecture designed for real-time target tracking on microcontrollers (MCUs) with as little as 512KB of RAM. The system combines efficient object detection using a FOMO-based approach with temporal sequence processing through an RNN head to predict camera motion for keeping targets in frame.

## Key Features

- **Two-Stage Architecture**: Combines MobileNetV2-FOMO detector with lightweight RNN head
- **Real-Time Performance**: Optimized for deployment on resource-constrained devices like ESP32-S3
- **Handball Player Tracking**: Specifically designed for tracking players in handball matches
- **Supercomputer Training**: Models trained on Leonardo EuroHPC infrastructure
- **Grayscale Support**: Optional grayscale processing for reduced computational requirements

## Architecture Overview

MCUMotionNet consists of two main components:

1. **Stage 1 - FOMO-based Person Detector**: Uses a truncated MobileNetV2 backbone to detect player centroids in each frame
2. **Stage 2 - RNN Motion Prediction Head**: Processes temporal sequences of features to predict camera motion

The complete architecture is detailed in our paper (see `docs/MCUMotionNet_Paper.tex`).

## Project Structure

```
├── data/                    # Dataset directories (not included in repo)
├── docs/                    # Documentation and research papers
├── logs/                    # Training logs and TensorBoard data
├── preparation/             # Data preparation scripts
├── results/                 # Experimental results
├── download_weight.py       # Download MobileNetV2 weights
├── extract_tensorboad_data.py # Parse TensorBoard logs
├── fomo_trainer.py          # FOMO model training utilities
├── fomo_trainer_grayscale.py # Grayscale FOMO training
├── load.sh                  # Environment setup script for Leonardo supercomputer
├── simulate_camera_movement.py # Camera movement simulation
├── simulate_person_detector.py # Person detection simulation
├── simulate_person_detector_grayscale.py # Grayscale detection simulation
├── train_combined_model_stage2.py # Stage 2 training script
├── train_person_detector.py # Stage 1 training script (RGB)
├── train_person_detector_grayscale.py # Grayscale Stage 1 training
└── visualize_annotated_sequences.py # Visualization utilities
```

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd MCUMotionNet
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Download MobileNetV2 weights:
```bash
python download_weight.py
```

## Dataset

The training dataset consists of handball video sequences with camera movement annotations. The dataset is available on Hugging Face:

**Dataset URL**: https://huggingface.co/datasets/scampion/handball_video_sequences

To download and use the dataset, you can use the Hugging Face `datasets` library:

```python
from datasets import load_dataset

# Load the dataset
dataset = load_dataset("scampion/handball_video_sequences")
```

Or use git to clone the repository:

```bash
git clone https://huggingface.co/datasets/scampion/handball_video_sequences
```

## Training

### Stage 1: FOMO Person Detector Training (RGB)
```bash
python train_person_detector.py
```

### Stage 1: FOMO Person Detector Training (Grayscale)
```bash
python train_person_detector_grayscale.py
```

### Stage 2: RNN Head Training
```bash
python train_combined_model_stage2.py --num-videos 1000 --experiment-name all
```

Available experiment configurations:
- extra_small_rnn_lstm_filters_2_dense_4
- small_rnn_lstm_filters_4_dense_8
- medium_rnn_lstm_filters_8_dense_16
- large_rnn_lstm_filters_16_dense_16
- extra_large_rnn_lstm_filters_32_dense_32

## Simulation and Testing

Run person detection simulation (RGB):
```bash
python simulate_person_detector.py
```

Run person detection simulation (Grayscale):
```bash
python simulate_person_detector_grayscale.py
```

Simulate camera movement prediction:
```bash
python simulate_camera_movement.py --video-path <path-to-video> --model-path <path-to-model>
```

Visualize annotated sequences:
```bash
python visualize_annotated_sequences.py <path-to-video>
```

## Leonardo Supercomputer Setup

To run on the Leonardo EuroHPC supercomputer:
```bash
source load.sh
```

## TensorBoard Data Extraction

Extract training metrics from TensorBoard logs:
```bash
python extract_tensorboad_data.py
```

## Deployment

To deploy on ESP32-S3:

1. Quantize the trained model using TensorFlow Lite
2. Convert to a format compatible with your microcontroller framework
3. Flash the model and inference code to the device

## Results

Training results and model performance metrics are stored in the `results/` directory. Use TensorBoard to visualize training progress:

```bash
tensorboard --logdir logs/
```

## Citing This Work

If you use MCUMotionNet in your research, please cite our paper:

```bibtex
@article{campion2024mcumotionnet,
  title={MCUMotionNet: A Lightweight, Two-Stage Architecture for Real-Time Object Tracking on Memory-Constrained Microcontrollers},
  author={Campion, Sébastien},
  journal={arXiv preprint},
  year={2024}
}
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- This research was supported by the Leonardo EuroHPC supercomputer infrastructure
- Sport.video platform for providing handball match footage
- Edge Impulse for the FOMO architecture inspiration

## Contact

For questions and contributions, please contact Sébastien Campion or open an issue in this repository.
