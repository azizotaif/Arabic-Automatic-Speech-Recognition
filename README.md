# Arabic Automatic Speech Recognition

This End-to-End Arabic automatic speech recognition system consists of two models, an acoustic model which is trained using wav2vec 2.0 framework, and a 4-gram language model which is trained using KenLM toolkit

## Datasets
The acoustic model is trained and tested on a combination of two datasets comprising 1157 hours in total
- [ArabicSpeech's MGB-2 Dataset](https://arabicspeech.org/mgb2)
- [Mozilla's Common Voice Arabic Dataset](https://commonvoice.mozilla.org/en/datasets)

The language model is trained on 8 million Arabic sentances collected from the [arabic_billion_words](https://huggingface.co/datasets/arabic_billion_words) dataset from huggingface
## Trained Models
[Download Acoustic Model (360.2 MB)](https://drive.google.com/file/d/1KHAOrkzGZpQtwsD5_7ACN-rk1NA7koMp/view?usp=sharing)

[Download Language Model (9.02 GB)](https://drive.google.com/file/d/1jYp6ZkZzNcikJ_WEKduNLmiiF-qzEfnR/view?usp=sharing)

## Performance

| Word Error Rate (WER) | Character Error Rate (CER) |
|:-----------------------:|:----------------------------:|
| 17.8%                 | 9.5%                       |


## Deploy

- Clone this repository and install the requirements
``` shell
git clone https://github.com/azizotaif/Arabic-Automatic-Speech-Recognition.git
pip install -r Arabic-Automatic-Speech-Recognition/requirements.txt
```

- Download both the [acoustic](https://drive.google.com/file/d/1KHAOrkzGZpQtwsD5_7ACN-rk1NA7koMp/view?usp=sharing) and [language model](https://drive.google.com/file/d/1jYp6ZkZzNcikJ_WEKduNLmiiF-qzEfnR/view?usp=sharing) and place them inside the models directory
``` shell
Arabic-Automatic-Speech-Recognition/models/
```
Make sure they are names as follow:

Acoustic model : pytorch_model.bin

Language model : language_model.arpa

- Run the Flask web app and pass a port number as argument, if no port is given, the default port 5000 will be used
``` shell
python app.py --port 7000
```

- Open a web browser and enter your IP and the port specified in the previous step
Example:
``` shell
192.168.1.100:7000
```

- The following page will appear, upload an audio file and click TRANSCRIBE

![image](https://drive.google.com/uc?export=view&id=1aDOkDyk3fjsDzu8iz8TBQhi2QHtFswAq)

![image](https://drive.google.com/uc?export=view&id=1TM3I484z5SXzWAg3sRbohwZBgtKXSlm3)

## 🔗 Contact me
[![Email](https://img.shields.io/badge/Gmail-D14836?style=for-the-badge&logo=gmail&logoColor=white)](mailto:otaif.abdulaziz@gmail.com)
[![Linkedin](https://img.shields.io/badge/linkedin-0A66C2?style=for-the-badge&logo=linkedin&logoColor=white)](https://linkedin.com/in/abdulazizotaif)
