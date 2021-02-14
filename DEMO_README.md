# Gesture recognition demo
The demo shows how to perform prediction gestures using web camera.

## Prerequisits
Currently demo tested only on Windows 10 and Linux Ububntu 18 platforms. For running demo on Respbery Pi  installation process can be different and will be defined later.
1. Install requirements
```bash
pip install requirements.txt
```
Please carefully read [instruction](https://docs.openvinotoolkit.org/latest/openvino_docs_install_guides_installing_openvino_pip.html) how to install and start work with openvino (initialize enviroment) from this page. It may be useful for continue work.

2. Downloading models
Prepared for running models can be found on [Google Drive](https://drive.google.com/drive/folders/1CRfoX2bIBNEKTeT9WrioOGO4otE9LyDK?usp=sharing)
Model converted using OpenVINO 2021.2 and Open Model Zoo.

## Run demo

```bash
python main.py -m_a common-sign-language-0001.xml -m_d person-detection-asl.xml -i <camera_id> -d CPU
```

## TO DO:
- [ ] installation way for Raspberry Pi
- [ ] add ONNX RunTime backend
- [ ] handle false positives
