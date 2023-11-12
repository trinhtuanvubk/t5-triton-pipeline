import json
import numpy as np
import torch
from torch import autocast
from pathlib import Path
import random

import triton_python_backend_utils as pb_utils

from torch.utils.dlpack import to_dlpack, from_dlpack

from transformers import T5ForConditionalGeneration, AutoTokenizer
from loguru import logger


T5_CONFIG = "config/t5_config"

cur_folder = Path(__file__).parent

t5_tokenizer_path = str(cur_folder/T5_CONFIG)



class TritonPythonModel:
    def initialize(self, args):
        # parse model_config
        self.model_config = model_config = json.loads(args["model_config"])
        # get last configuration
        last_output_config = pb_utils.get_output_config_by_name(model_config, "paraphrase_answer")
        #convert triton type to numpy type/
        self.last_output_dtype = pb_utils.triton_string_to_numpy(last_output_config["data_type"])

        # t5 tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(t5_tokenizer_path, local_files_only=True)
        # string dtype
        self._dtypes = [np.bytes_, np.object_]

    def execute(self, requests):
        responses = []
        for request in requests:
           
            input_text = pb_utils.get_input_tensor_by_name(request, "input_text")
            input_text = input_text.as_numpy().astype(np.bytes_)[0]
            input_text = [i.decode("utf-8").lower() for i in input_text]
           
            num_return_sequence = 10
            batch = self.tokenizer(['Paraphrasing this sentence: ' + answer for answer in input_text],
                    truncation=True,
                    padding='longest',
                    max_length=256,
                    return_tensors="pt")
            
            input_ids = batch.input_ids
            attention_mask = batch.attention_mask

            input_ids = pb_utils.Tensor(
                "input_ids",
                input_ids.numpy().astype(np.int64)
            )
            
            attention_mask = pb_utils.Tensor(
                "attention_mask",
                attention_mask.numpy().astype(np.int64)
            )

            t5_request = pb_utils.InferenceRequest(
                model_name="t5",
                requested_output_names=["output_ids"],
                inputs=[input_ids, attention_mask],
            )

            response = t5_request.exec()

            t5_ids = pb_utils.get_output_tensor_by_name(response, "output_ids")
            t5_ids = from_dlpack(t5_ids.to_dlpack()).clone()

            # # print(model.generate.__dict__)
            # translated = t5_model.generate(**batch,
            #                     max_length=256,
            #                     num_beams=10,
            #                     num_return_sequences=num_return_sequence,
            #                     temperature=1.3)

            # logger.debug(f"t5ids: {t5_ids.shape}")
                                
            # num_sentences = len(answers)
            all_tgt_texts = [self.tokenizer.batch_decode(t5_ids[i*num_return_sequence:(i+1)*num_return_sequence], skip_special_tokens=True) for i in range(len(input_text))]
            
            # final_anwers = [random.choice(answer) for answer in all_tgt_texts]
            # final_anwers = all_tgt_texts

            
            # ==== Sending Response ====
            last_output_tensor = pb_utils.Tensor(
                "paraphrase_answer", np.array([[i.encode('utf-8') for i in final_answers] for final_answers in all_tgt_texts], dtype=self._dtypes[0]))
            
            inference_response = pb_utils.InferenceResponse([last_output_tensor])
        
            responses.append(inference_response)

            return responses



