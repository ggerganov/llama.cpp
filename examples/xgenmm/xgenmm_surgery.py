import torch
import argparse
from open_flamingo import create_model_and_transforms
from omegaconf import OmegaConf
import os
import time

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt_pth", type=str, default='/export/share/manli_shu/models/open-flamingo-dev/anyres_ablation_HFSiglip_patch128-kosmos_non_instruct-phi3_4k_instruct_nq128_pre_V3_5-llava_1p6_ocrmathmix_v4-8x8-ckpt2/checkpoint_0.pt')
    parser.add_argument('--save_pth', type=str, default='/export/share/yutong/xgenmm/llamacpp_wd')
    parser.add_argument('--version', type=str, default='siglip_kosmos_phi3_4k_instruct', help='help identify the version of the saved ckpt')
    return parser.parse_args()

VISION_ENCODER_KEY = 'vision_encoder'
LLM_KEY = 'lang_model'
PROJECTOR = 'vision_tokenizer'


if __name__ == "__main__":
    # load ckpt
    args = get_args()
    print("🟡 Loading ckpt...")
    start = time.time()
    ckpt = torch.load(args.ckpt_pth)["model_state_dict"]
    end = time.time()
    print(f"🟢 time used: [{end-start:.3f} s] | Done with loading ckpt")
    
    # sanity check
    unexpected_component_keys = set()
    for k in list(ckpt.keys()):
        matched = False
        for c in ['vision_encoder', 'lang_model', 'vision_tokenizer']:
            if k.startswith(c):
                matched = True
                continue
        if not matched:
            unexpected_component_keys.add(k)
            
    if len(unexpected_component_keys) > 0:
        print(f"❗❗❗ Unexpected component keys: {unexpected_component_keys}. Proceed with caution.")
    
    save_dir = f"{args.save_pth}/{args.version}"
    print("🟡 Instaiate the model.")
    start = time.time()
    cfg = dict(
        model_family = 'kosmos',
        lm_path = 'microsoft/Phi-3-mini-4k-instruct',
        vision_encoder_path = 'google/siglip-so400m-patch14-384',
        vision_encoder_pretrained = 'google',
        num_vision_tokens = 128,
        image_aspect_ratio = 'anyres',
        anyres_patch_sampling = True,
        anyres_grids=[[1,2],[2,1],[2,2],[3,1],[1,3]],
        ckpt_pth = args.ckpt_pth)
    cfg = OmegaConf.create(cfg)
    if cfg.model_family in ['kosmos-instruct', 'kosmos', 'llava']:
        additional_kwargs = {
            "image_aspect_ratio": cfg.image_aspect_ratio,
            }
        if cfg.model_family in ['kosmos-instruct', 'kosmos']:
            additional_kwargs.update({
                "num_vision_tokens": cfg.num_vision_tokens,
                "anyres_patch_sampling": cfg.anyres_patch_sampling,
            })
    model, image_processor, tokenizer = create_model_and_transforms(
                                        clip_vision_encoder_path=cfg.vision_encoder_path,
                                        clip_vision_encoder_pretrained=cfg.vision_encoder_pretrained,
                                        lang_model_path=cfg.lm_path,
                                        tokenizer_path=cfg.lm_path,
                                        model_family=cfg.model_family,
                                        **additional_kwargs)
    print(model)
    exit(1)
    model.load_state_dict(ckpt, strict=True)
    end = time.time()
    print(f"🟢 time used: [{end-start:.3f} s] | Done with instaiating the model.")
    

    print("🟡 Peforming the surgery...")
    
    model.lang_model.save_pretrained(f"{save_dir}/llm")    
    
    model.vision_encoder.config.save_pretrained(f"{save_dir}/vision_encoder")
    vision_encoder_tensors = {k.split(VISION_ENCODER_KEY + '.')[-1]: v.float() for k, v in ckpt.items() if k.startswith(VISION_ENCODER_KEY)}
    save_path = f"{save_dir}/vision_encoder/xgenmm.vision_encoder"
    torch.save(vision_encoder_tensors, save_path)
    
    
    projector_tensors = {k.split(PROJECTOR + '.')[-1]: v.float() for k, v in ckpt.items() if k.startswith(PROJECTOR)}
    save_path = f"{save_dir}/xgenmm.projector"
    torch.save(projector_tensors, save_path)
    
    # processors
    tokenizer.save_pretrained(f"{save_dir}/tokenizer")
    # will hard code the image_processor in the convert_image_encoder_to_gguf.py
    
    end = time.time()
    print(f"🟢 time used: [{end-start:.3f} s]")