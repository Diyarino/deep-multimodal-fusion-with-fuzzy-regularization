# Multimodal Synthetic Dataset Balancing

Deep neural networks excel at processing **unimodal data** (sensors, images, audio), but real-world **multimodal systems** often face corrupted signals (sensor failures, noise, or coverage gaps). This repository contains a fuzzy regularization method** for deep multimodal networks that:  

- Dynamically adjusts feature importance based on **signal quality**  
- Compensates for sensor malfunctions/uncertainties  
- Works with **image + time-series data**  


### ✨ Fuzzy latent-space regularization


In the realm of neural networks, **target activations** represent the desired output values for specific neurons or layers, often reflecting the ground truth or a predefined ideal state. The primary objective of a loss function is to quantify the discrepancy between the network's actual activations and the target activations. By minimizing this loss, we guide the network to adjust its internal parameters (weights and biases) such that its computed activations increasingly approximate the desired targets.

![screenshot](traget.png)


As training progresses over multiple epochs, the loss function continually guides the weight adjustments. This iterative process allows the neuron's activations to gradually "converge" towards the predefined target activations for each class, thereby improving the network's ability to detect anomalies.

![til](animation.gif)


### 📦 Prerequisites

Before you begin, ensure your environment meets the following requirements:

* **Python** ≥ 3.6
* **PyTorch** ≥ 1.0 (CUDA support recommended for faster training)

We also recommend using a virtual environment (e.g., `venv` or `conda`) to avoid package conflicts.

## 🔗 Multimodal Robot Kinematic Datasets

This repository provides access to three multimodal robot movement datasets, each including at a minimum the **camera** and **kinematics** modalities. For detailed descriptions of the datasets, data collection procedures, and experimental use cases, please refer to our paper [2]:

**"Performance benchmarking of multimodal data-driven approaches in industrial settings"** – [Link to Paper](https://www.sciencedirect.com/science/article/pii/S266682702500074X?via%3Dihub)

1. **MuJoCo: UR5 Robot Motion** – [Link to Dataset 1](https://zenodo.org/records/14041622)
2. **ABB Studio: Single Robot Welding Station** – [Link to Dataset 2](https://zenodo.org/records/14041488)
3. **ABB Studio: Dual Robot Welding Station** – [Link to Dataset 3](https://zenodo.org/records/14041416)

Each dataset captures robot motion across various tasks and environments, providing synchronized data streams for machine learning and robotics research.

## 📌 Citation
If you use this code or build upon our work, please cite our paper:


```bibtex
@inproceedings{altinses2023deep,
  title={Deep Multimodal Fusion with Corrupted Spatio-Temporal Data Using Fuzzy Regularization},
  author={Altinses, Diyar and Schwung, Andreas},
  booktitle={IECON 2023-49th Annual Conference of the IEEE Industrial Electronics Society},
  pages={1--7},
  year={2023},
  organization={IEEE}
}
```


## 📚 Related Projects 

This project builds on concepts from multimodal representation learning, attention-based fusion, and anomaly detection in industrial systems. Below are selected related works and projects that inspired or complement this research:

<a id="1">[1]</a> Altinses, D., & Schwung, A. (2023, October). Multimodal Synthetic Dataset Balancing: A Framework for Realistic and Balanced Training Data Generation in Industrial Settings. In IECON 2023-49th Annual Conference of the IEEE Industrial Electronics Society (pp. 1-7). IEEE.

<a id="2">[2]</a> Altinses, D., & Schwung, A. (2025). Performance benchmarking of multimodal data-driven approaches in industrial settings. Machine Learning with Applications, 100691.

<a id="3">[3]</a> Altinses, D., & Schwung, A. (2023, October). Deep Multimodal Fusion with Corrupted Spatio-Temporal Data Using Fuzzy Regularization. In IECON 2023-49th Annual Conference of the IEEE Industrial Electronics Society (pp. 1-7). IEEE.

<a id="3">[4]</a> Altinses, D., Torres, D. O. S., Lier, S., & Schwung, A. (2025, February). Neural Data Fusion Enhanced PD Control for Precision Drone Landing in Synthetic Environments. In 2025 IEEE International Conference on Mechatronics (ICM) (pp. 1-7). IEEE.

<a id="3">[5]</a> Torres, D. O. S., Altinses, D., & Schwung, A. (2025, March). Data Imputation Techniques Using the Bag of Functions: Addressing Variable Input Lengths and Missing Data in Time Series Decomposition. In 2025 IEEE International Conference on Industrial Technology (ICIT) (pp. 1-7). IEEE.



