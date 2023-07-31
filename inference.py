from PIL import Image, ImageDraw
import numpy as np
import torch
import torch.nn.functional as F
from typing import List
from tqdm.notebook import tqdm

from xattention_guidance import ptp_utils
from xattention_guidance.xattention_guidance import compute_ca_loss, AttentionStore


def draw_box(pil_img, bboxes, phrases, save_path):
    draw = ImageDraw.Draw(pil_img)
    phrases = [x.strip() for x in phrases.split(';')]
    for obj_bboxes, phrase in zip(bboxes, phrases):
        for obj_bbox in obj_bboxes:
            x_0, y_0, x_1, y_1 = obj_bbox[0], obj_bbox[1], obj_bbox[2], obj_bbox[3]
            draw.rectangle([int(x_0 * 512), int(y_0 * 512), int(x_1 * 512), int(y_1 * 512)], outline='red', width=5)
            draw.text((int(x_0 * 512) + 5, int(y_0 * 512) + 5), phrase, fill=(255, 0, 0))
    pil_img.save(save_path)


def phrase2idx(prompt, phrases):
    phrases = [x.strip() for x in phrases.split(';')]
    prompt_list = prompt.strip('.').split(' ')
    object_positions = []
    for obj in phrases:
        obj_position = []
        for word in obj.split(' '):
            obj_first_index = prompt_list.index(word) + 1
            obj_position.append(obj_first_index)
        object_positions.append(obj_position)

    return object_positions


def aggregate_attention(prompts: List[str], attention_store: AttentionStore, res: int, from_where: List[str], is_cross: bool, select: int):
    out = []
    attention_maps = attention_store.get_average_attention(mask=True)
    num_pixels = res ** 2
    for location in from_where:
        for item in attention_maps[f"{location}_{'cross' if is_cross else 'self'}"]:
            if item.shape[1] == num_pixels:
                cross_maps = item.reshape(len(prompts), -1, res, res, item.shape[-1])[select]
                out.append(cross_maps)
    out = torch.cat(out, dim=0)
    out = out.sum(0) / out.shape[0]
    return out.cpu()


def show_cross_attention(tokenizer, prompts: List[str], attention_store: AttentionStore, res: int, from_where: List[str], select: int = 0):
    tokens = tokenizer.encode(prompts[select])
    decoder = tokenizer.decode
    attention_maps = aggregate_attention(prompts, attention_store, res, from_where, True, select)
    images = []
    for i in range(len(tokens)):
        image = attention_maps[:, :, i]
        image = 255 * image / image.max()
        image = image.unsqueeze(-1).expand(*image.shape, 3)
        image = image.numpy().astype(np.uint8)
        image = np.array(Image.fromarray(image).resize((256, 256)))
        image = ptp_utils.text_under_image(image, decoder(int(tokens[i])))
        images.append(image)
    ptp_utils.view_images(np.stack(images, axis=0))


def bbox_inference(
    device,
    model,
    controller,
    prompt,
    bboxes,
    phrases,
    height=512,
    width=512,
    timesteps=51,
    loss_scale=30,
    guidance_scale=7.5
):
    # Get Cross-Attentions
    ptp_utils.register_attention_control(model, controller)

    reset_mask = not controller.sum_blocks[1]

    # Get Object Positions
    object_positions = phrase2idx(prompt, phrases)

    # Encode Classifier Embeddings
    uncond_input = model.tokenizer(
        [""] * 1, padding="max_length", max_length=model.tokenizer.model_max_length, return_tensors="pt"
    )
    uncond_embeddings = model.text_encoder(uncond_input.input_ids.to(device))[0]

    # Encode Prompt
    input_ids = model.tokenizer(
            [prompt] * 1,
            padding="max_length",
            truncation=True,
            max_length=model.tokenizer.model_max_length,
            return_tensors="pt",
        )

    cond_embeddings = model.text_encoder(input_ids.input_ids.to(device))[0]
    text_embeddings = torch.cat([uncond_embeddings, cond_embeddings])
    generator = torch.manual_seed(42)  # Seed generator to create the inital latent noise

    _, latents = ptp_utils.init_latent(None, model, height=height, width=width, generator=generator, batch_size=1)

    model.scheduler.set_timesteps(timesteps)

    latents = latents * model.scheduler.init_noise_sigma

    loss = torch.tensor(10000)

    for index, t in enumerate(tqdm(model.scheduler.timesteps)):
        iteration = 0
        controller.set_to_mask(False)
        while loss.item() / 30 > 0.2 and iteration < 5 and index < 10:
            latents = latents.requires_grad_(True)
            latent_model_input = latents
            latent_model_input = model.scheduler.scale_model_input(latent_model_input, t)
            noise_pred = model.unet(latent_model_input, t, encoder_hidden_states=cond_embeddings)["sample"]

            # update latents with guidance
            loss = compute_ca_loss(controller.get_average_attention(), bboxes=bboxes, object_positions=object_positions) * loss_scale
            grad_cond = torch.autograd.grad(loss.requires_grad_(True), [latents])[0]

            latents = latents - grad_cond * model.scheduler.sigmas[index] ** 2
            iteration += 1
            torch.cuda.empty_cache()
            controller.reset(mask=reset_mask)

        controller.set_to_mask(True)
        with torch.no_grad():
            latent_model_input = torch.cat([latents] * 2)

            latent_model_input = model.scheduler.scale_model_input(latent_model_input, t)
            noise_pred = model.unet(latent_model_input, t, encoder_hidden_states=text_embeddings)["sample"]

            # perform guidance
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

            latents = model.scheduler.step(noise_pred, t, latents).prev_sample
            latents = controller.step_callback(latents)
            torch.cuda.empty_cache()

            # Store last map for cross-attenting masking
            if index < len(model.scheduler.timesteps) - 1:
                controller.reset(mask=reset_mask)

    images = ptp_utils.latent2image(model.vae, latents)
    pil_images = [Image.fromarray(image) for image in images]
    return pil_images


def per_box_image(
    device,
    model,
    box_prompt,
    background_prompt,  
    bbox,
    mask_method="threshold",
    height=512,
    width=512,
    timesteps=51,
    loss_scale=30,
    guidance_scale=7.5
):
    prompt = background_prompt + "with " + box_prompt
    phrase = "with " + box_prompt

    controller = AttentionStore(sum_blocks=(False, True))

    pil_images = bbox_inference(
        device=device,
        model=model,
        controller=controller,
        prompt=prompt,
        bboxes=bbox,
        phrases=phrase,
        height=height,
        width=width,
        timesteps=timesteps,
        loss_scale=loss_scale,
        guidance_scale=guidance_scale
    )
    
    # Fetch Cross-Attention map for object token
    # assume that phrase ends with object token; get last occurrence of token
    obj_idx = phrase2idx(prompt, box_prompt.strip().split()[-1])[-1]
    attention_maps = aggregate_attention(
        [prompt],
        controller,
        res=16,
        from_where=('up', 'down')
    )
    xattn_map = attention_maps[:, :, obj_idx]

    xattn_mask = F.interpolate(
        xattn_map.view(1, 1, xattn_map.size(0), xattn_map.size(1)),
        (height // 8, width // 8),
        mode='bicubic'
    )

    if mask_method == "threshold":
        xattn_hist = torch.histogram(xattn_mask.flatten(), 100)
        threshold_bin = torch.where((xattn_hist.hist.cumsum(0) / xattn_mask.numel()) > 0.95)[0][0]
        threshold = xattn_hist.bin_edges[threshold_bin]
        xattn_mask = torch.where(xattn_mask > threshold, 1., 0.)
    else:
        raise NotImplementedError
    
    return pil_images[0], xattn_mask