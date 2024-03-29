# CLEM-REG CLI Napari Plugin

## Introduction

This repository contains the command-line interface (CLI) version of the CLEM-REG Napari plugin. CLEM-REG is a powerful tool for registering and analyzing correlative light and electron microscopy (CLEM) images. This CLI version allows you to use the same functionalities from the command line, making it easier to integrate with other tools and automate your workflows. This has been tested on the Liverpool Shared Research Facilities HPC Cluster and we have found a significant perfomance increase by running the registration without napari. The next steps are to integrate more segmentation algorithms and to give users a web interface to visualise intermediary results.

## Installation

You can install the CLI version of the CLEM-REG Napari plugin by cloning this repository and building the container from the .def file. Here's how you can do it:

```bash
sudo apptainer build cbf_clem_reg.sif cbf_clem_reg.def
```
## Usage

You can use the CLI version of the CLEM-REG Napari plugin by running the container with your images and parameters as command-line arguments. Here’s an example:

```bash
apptainer run --containall --bind ./:/input,output/:/output --nv cbf_clem_reg.sif --lm_input /input/{LM_FILE} --em_input /input/{EM_FILE} --registration_algorithm 'Rigid CPD'
```
## TODO
- Implement alternate segmentation algorithms
- Web interface for intermediary results
- SLURMGUI workflow
