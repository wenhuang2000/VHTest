import argparse
import torch

from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import process_images, tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria

from PIL import Image

import requests
from PIL import Image
from io import BytesIO
from transformers import TextStreamer

import os
import pandas as pd
import glob

vh_mode_list = ["counting", "shape", "color", "orientation", "size", "position", "OCR", "existence"]

import numpy as np
import random
def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    print("random seed:", seed)
     

def load_image(image_file):
    if image_file.startswith('http://') or image_file.startswith('https://'):
        response = requests.get(image_file)
        image = Image.open(BytesIO(response.content)).convert('RGB')
    else:
        image = Image.open(image_file).convert('RGB')
    return image


def main(args):
    TEST_XLSX_PATH = args.test_xlsx_path
    IMAGE_PATH = args.image_path
    RESULTS_XLSX_PATH = args.results_xlsx_path
    
    print(f"will save results to {RESULTS_XLSX_PATH}")

    if not os.path.exists(RESULTS_XLSX_PATH):
        os.makedirs(RESULTS_XLSX_PATH)
    
    setup_seed(42)
    disable_torch_init()

    model_name = get_model_name_from_path(args.model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(args.model_path, args.model_base, model_name, args.load_8bit, args.load_4bit, device=args.device)

    if 'llama-2' in model_name.lower():
        conv_mode = "llava_llama_2"
    elif "v1" in model_name.lower():
        conv_mode = "llava_v1"
    elif "mpt" in model_name.lower():
        conv_mode = "mpt"
    else:
        conv_mode = "llava_v0"

    if args.conv_mode is not None and conv_mode != args.conv_mode:
        print('[WARNING] the auto inferred conversation mode is {}, while `--conv-mode` is {}, using {}'.format(conv_mode, args.conv_mode, args.conv_mode))
    else:
        args.conv_mode = conv_mode

    conv = conv_templates[args.conv_mode].copy()
    if "mpt" in model_name.lower():
        roles = ('user', 'assistant')
    else:
        roles = conv.roles



    for vh_mode in vh_mode_list:

        pattern = os.path.join(TEST_XLSX_PATH, f"{vh_mode}*.xlsx")
        for file_path in glob.glob(pattern):
            if os.path.exists(file_path):


                image_base_path = f"{IMAGE_PATH}/{vh_mode}/"

                df = pd.read_excel(file_path)

                file_name_without_extension = os.path.splitext(os.path.basename(file_path))[0]
                results_xlsx_path = os.path.join(RESULTS_XLSX_PATH, file_name_without_extension + "_query_LLaVA13bv1.5.xlsx")
                print(f'vh_mode: {vh_mode} output_file: {results_xlsx_path}')

                df.to_excel(results_xlsx_path, index=False)


                if 'LLaVA_bug_success' in df.columns:
                    idx = df.columns.get_loc('LLaVA_bug_success') + 1
                    df.insert(idx, 'LLaVA_response', '')
                    df.insert(idx + 1, 'LLaVA_bug_success', '')
                else:
                    df['LLaVA_response'] = ''
                    df['LLaVA_bug_success'] = ''

                for index, row in df.iterrows():
                    image_file = os.path.join(image_base_path, row['image_path'])
                    question = row['question']

                    # Reset the conversation object for each iteration
                    conv = conv_templates[args.conv_mode].copy()

                    image = load_image(image_file)
                    image_tensor = process_images([image], image_processor, model.config)
                    if type(image_tensor) is list:
                        image_tensor = [image.to(model.device, dtype=torch.float16) for image in image_tensor]
                    else:
                        image_tensor = image_tensor.to(model.device, dtype=torch.float16)

                    if model.config.mm_use_im_start_end:
                        inp = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + question
                    else:
                        inp = DEFAULT_IMAGE_TOKEN + '\n' + question
                    conv.append_message(conv.roles[0], inp)
                    image = None
                    conv.append_message(conv.roles[1], None)


                    prompt = conv.get_prompt()

                    input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).to(model.device)
                    stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
                    keywords = [stop_str]
                    stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)
                    streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

                    with torch.inference_mode():
                        output_ids = model.generate(
                            input_ids,
                            images=image_tensor,
                            do_sample=True if args.temperature > 0 else False,
                            temperature=args.temperature,
                            max_new_tokens=args.max_new_tokens,
                            streamer=streamer,
                            use_cache=True,
                            stopping_criteria=[stopping_criteria])

                    outputs = tokenizer.decode(output_ids[0, input_ids.shape[1]:]).strip()
                    print(f"index: {index}, outputs: {outputs}")
                    df.at[index, 'LLaVA_response'] = outputs


                    if index % 5 == 0: 
                        df.to_excel(results_xlsx_path, index=False)


                df.to_excel(results_xlsx_path, index=False)




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="facebook/opt-350m")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--image-file", type=str)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--conv-mode", type=str, default=None)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--load-8bit", action="store_true")
    parser.add_argument("--load-4bit", action="store_true")
    parser.add_argument("--debug", action="store_true")

    parser.add_argument("--test-xlsx-path", default=None)
    parser.add_argument("--image-path", default=None)
    parser.add_argument("--results-xlsx-path", default=None)

    args = parser.parse_args()
    main(args)
