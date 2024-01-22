# PhD Thesis: [Observe, Predict, Adapt: A Neural model of Adaptive Motor Control]

## Abstract
Biological control systems have evolved to perform efficiently in an environment characterized by high uncertainty and unexpected disturbances, while relying on noisy sensors and unreliable actuators.  Despite these challenges, biological control systems remain superior to engineered control systems in many respects. This edge in performance can be attributed to the exceptional ability of the brain to predict adaptively, and continuously update its control strategies in the face of uncertainties. Consequently, to harness these control abilities, it is crucial to delve into the study and modeling of cortical functioning.

This thesis presents a novel and comprehensive approach to elucidate the underlying mechanisms governing motor control. Specifically, we propose a biologically plausible spiking neural network model of the primate sensory-motor control system. The core of the model lies in its effective handling of noisy observations, integration of different sensory modalities, and the ability to learn to control the arm in the presence of perturbations. This is accomplished through the introduction of the Neural Adaptive Filter, a mechanism that dynamically predicts sensory consequences based on control inputs and observations.

The developed functional model of the sensory-motor control system exhibits complex behaviours observed in primates' reaching and demonstrates neural activities comparable to experimental findings. By adopting a spiking architecture, and connecting the lower-level synaptic dynamics and higher-level behaviours, such as visuomotor rotation, the model offers valuable insights into underlying mechanisms. Furthermore, the incorporation of anatomical structure and neural constraints enhances the biological plausibility and explanatory power of the model.

Moreover, the realization of a functional spiking model of the sensory-motor control system holds broader implications, particularly in control theory and its applications. This spiking model preserves the brain's inherent sparse coding, optimal performance, and energy efficiency, all of which are highly advantageous for engineering solutions. The model's operation can be generalized to that of a filter-controller framework capable of adapting to unknown nonlinear systems. This enhances robustness and plasticity derived from biological inspiration. Ultimately, by integrating the principles of a biological control system into modern control theory, our model not only offers insights into the sensory-motor control system and proposes potential advancements in modern control methodologies.

## Repository Structure
This repository contains the code and data associated with the thesis titled "[Your Thesis Title]". Here's a quick overview of the repository's structure:

- `Lorenz/`:  Jupyter notebook for the Lorenz Attractor system
- `Pendulum/`: Includes jupyter notebooks and py files for the damped control pendulum
- `Twolink and VMR/`: Jupyter notebooks and files for both the two link arm and visuomotor experiments, data analysis, etc.
- `docs/`: Additional documentation and supporting materials.

## Getting Started
### Prerequisites
Nengo version: 3.1.0
Python version: 3.7.11 

## Link to thesis
This will be updated to provide the link
