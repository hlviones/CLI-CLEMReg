# CLEM-REG CLI Napari Plugin

## Introduction

This repository contains the command-line interface (CLI) version of the CLEM-REG Napari plugin. CLEM-REG is a powerful tool for registering and analyzing correlative light and electron microscopy (CLEM) images. This CLI version allows you to use the same functionalities from the command line, making it easier to integrate with other tools and automate your workflows.

## Features

The CLI version of the CLEM-REG Napari plugin provides the following features:

- **Image Registration**: Register your light microscopy (LM) and electron microscopy (EM) images with ease.
- **Transformation Matrix Application**: Apply the transformation matrix to your images for accurate correlation.
- **Visualization**: Visualize your registered images using Napari.

## Installation

You can install the CLI version of the CLEM-REG Napari plugin by cloning this repository and building the container from the .def file. Here's how you can do it:

```bash
sudo apptainer build cbf_clem_reg.sif cbf_Clem_reg.def
```
## Usage

You can use the CLI version of the CLEM-REG Napari plugin by running the container with your images and parameters as command-line arguments. Here’s an example:

```bash
apptainer run --containall --bind ./:/input,output/:/output --nv cbf_clem_reg.sif --lm_input /input/{LM_FILE} --em_input /input/{EM_FILE} --registration_algorithm 'Rigid CPD'
```
