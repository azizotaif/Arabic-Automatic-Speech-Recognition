from abc import ABC
import logging
import torch
import soundfile as sf
import librosa
import io
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC, Wav2Vec2CTCTokenizer
from ts.torch_handler.base_handler import BaseHandler
import ctcdecode
import numpy as np

logger = logging.getLogger(__name__)


class Wav2vecHandler(BaseHandler, ABC):
    """
    """
    def __init__(self):
        super(Wav2vecHandler, self).__init__()
        self.initialized = False

    def initialize(self, ctx):
        self.manifest = ctx.manifest

        properties = ctx.system_properties
        model_dir = properties.get("model_dir")
        self.device = torch.device("cpu")
        tokenizer = Wav2Vec2CTCTokenizer.from_pretrained(model_dir)

        # Loading language model
        vocab_dict = tokenizer.get_vocab()
        sort_vocab = sorted((value, key) for (key,value) in vocab_dict.items())
        vocab = [x[1].replace("|", " ") if x[1] not in tokenizer.all_special_tokens else "_" for x in sort_vocab]
        alpha = 2.5 # LM Weight
        beta = 0.0 # LM Usage Reward
        word_lm_scorer = ctcdecode.WordKenLMScorer('/home/ubuntu/webapp/model/4-gram.arpa', alpha, beta)
        self.decoder = ctcdecode.BeamSearchDecoder(
            vocab,
            num_workers=2,
            beam_width=128,
            scorers=[word_lm_scorer],
            cutoff_prob=np.log(0.000001),
            cutoff_top_n=40
        )

        # Loading acoustic model
        self.model = Wav2Vec2ForCTC.from_pretrained(model_dir)
        self.processor = Wav2Vec2Processor.from_pretrained(model_dir)
        self.model.to(self.device)
        self.model.eval()

        logger.debug('Transformer model from path {0} loaded successfully'.format(model_dir))

        self.initialized = True

    def preprocess(self, data):
        
        wav = data[0].get("data")
        if wav is None:
            wav = data[0].get("body")
        if wav is None:
            wav = data[0].get("file")
        
        speech_array, sampling_rate = sf.read(io.BytesIO(wav))
        if sampling_rate != 16000:
            speech_array, s = librosa.load(io.BytesIO(wav), sr=16000)

        inputs = speech_array
        
        return inputs

    def inference(self, inputs):
        with torch.no_grad():
            input_values = torch.tensor(inputs, device=self.device).unsqueeze(0)
            logits = self.model(input_values.float()).logits
            inference = self.decoder.decode_batch(logits.numpy())[0]

        return [inference]

    def postprocess(self, inference_output):
        print(inference_output[0])
        inference_output[0] = inference_output[0].encode('utf-8')
        print(inference_output[0])
        return inference_output


_service = Wav2vecHandler()


def handle(data, context):
    try:
        if not _service.initialized:
            _service.initialize(context)

        if data is None:
            return None

        data = _service.preprocess(data)
        data = _service.inference(data)
        data = _service.postprocess(data)

        return data
    except Exception as e:
        raise e