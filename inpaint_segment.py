import torch
from diffusers import AutoPipelineForInpainting
from transformers import AutoProcessor, AutoModelForSemanticSegmentation
from PIL import Image, ImageDraw
import time
import io

# --- Imports for the Web Server ---
import uvicorn
from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import StreamingResponse
import torch.nn.functional as F

# --- 1. Model Loading (Runs Once on Startup) ---
print("Loading all models. This will take a moment...")
ml_models = {}

# --- Load Inpainting Model ---
print("Loading Inpainting model (Kandinsky)...")
inpaint_pipeline = AutoPipelineForInpainting.from_pretrained(
    "kandinsky-community/kandinsky-2-2-decoder-inpaint", torch_dtype=torch.float16
)
inpaint_pipeline.to("cuda")
print("Compiling Inpainting UNet...")
inpaint_pipeline.unet = torch.compile(inpaint_pipeline.unet, mode="reduce-overhead", fullgraph=True)
ml_models["inpaint_pipeline"] = inpaint_pipeline
print("Inpainting model ready.")

# --- Load Semantic Segmentation Model (ClipSeg) ---
print("Loading Segmentation model (ClipSeg)...")
# Using the generic AutoProcessor and AutoModel classes for better compatibility
clipseg_processor = AutoProcessor.from_pretrained("CIDAS/clipseg-rd64-refined")
clipseg_model = AutoModelForSemanticSegmentation.from_pretrained("CIDAS/clipseg-rd64-refined", torch_dtype=torch.float16)
clipseg_model.to("cuda")
ml_models["segmentation_processor"] = clipseg_processor
ml_models["segmentation_model"] = clipseg_model
print("Segmentation model ready.")

print("\nAll models loaded. Server is ready to accept requests.")


# --- 2. FastAPI App and Endpoint Definitions ---
app = FastAPI()

# --- Endpoint for Inpainting (Unchanged) ---
@app.post("/inpaint/")
async def run_inpainting(
    prompt: str = Form(...),
    negative_prompt: str = Form(""),
    init_image_file: UploadFile = File(...),
    mask_image_file: UploadFile = File(...)
):
    """
    Receives an image, a mask, and prompts to perform inpainting.
    """
    print(f"Received inpainting request with prompt: '{prompt}'")
    start_time = time.time()

    init_image_bytes = await init_image_file.read()
    mask_image_bytes = await mask_image_file.read()

    init_image = Image.open(io.BytesIO(init_image_bytes)).convert("RGB")
    mask_image = Image.open(io.BytesIO(mask_image_bytes)).convert("RGB")

    pipe = ml_models["inpaint_pipeline"]
    image = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        image=init_image,
        mask_image=mask_image,
        num_inference_steps=20, # Added for potentially faster results
        strength=0.99
    ).images[0]

    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    buffer.seek(0)

    end_time = time.time()
    print(f"Inpainting took {end_time - start_time:.2f} seconds.")
    return StreamingResponse(buffer, media_type="image/png")

# --- MODIFIED Endpoint for Text-Based Segmentation Mask Creation ---
@app.post("/segment/")
async def run_segmentation_and_create_mask(
    text_prompt: str = Form(...),
    confidence_threshold: float = Form(0.4), # This threshold is now for the heatmap
    image_file: UploadFile = File(...)
):
    """
    Receives an image and a text prompt, runs ClipSeg to find the object,
    and returns a black and white MASK image based on the segmentation.
    """
    print(f"Received MASK creation request for '{text_prompt}' with threshold: {confidence_threshold}")
    start_time = time.time()

    # Read the original image
    image_bytes = await image_file.read()
    init_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    original_size = init_image.size

    # Get the segmentation model and processor
    processor = ml_models["segmentation_processor"]
    model = ml_models["segmentation_model"]

    # Prepare inputs for ClipSeg
    # The processor handles resizing and normalization
    inputs = processor(
        text=[text_prompt],
        images=[init_image],
        padding="max_length",
        return_tensors="pt"
    ).to("cuda")

    # Run the model
    with torch.no_grad():
        outputs = model(**inputs)

    # --- THIS IS THE KEY CHANGE FROM DINO TO CLIPSEG ---
    # 1. Get the raw prediction (logits) and resize it to the original image size
    # For ClipSeg, the output is in 'logits'.
    preds = outputs.logits.unsqueeze(1)
    resized_preds = F.interpolate(
        preds,
        size=original_size,
        mode='bilinear',
        align_corners=False
    )

    # 2. Process the prediction to create a binary mask
    # Apply sigmoid to get probabilities, then apply the threshold
    heatmap = torch.sigmoid(resized_preds.squeeze())
    binary_mask = (heatmap > confidence_threshold).cpu().numpy().astype('uint8') * 255

    # 3. Convert the numpy array mask to a PIL Image
    # 'L' mode is for 8-bit grayscale pixels (0=black, 255=white).
    mask_image = Image.fromarray(binary_mask, mode='L')
    # --- END OF CHANGE ---

    # 4. Save the MASK image to the buffer
    buffer = io.BytesIO()
    mask_image.save(buffer, format="PNG")
    buffer.seek(0)

    end_time = time.time()
    print(f"Created segmentation mask. Took {end_time - start_time:.2f} seconds.")
    return StreamingResponse(buffer, media_type="image/png")


# --- 3. Running the Server ---
if __name__ == "__main__":
    # It's good practice to bind to 0.0.0.0 to be accessible from other machines
    # on the network, not just the local machine.
    uvicorn.run(app, host="127.0.0.2", port=8008)
