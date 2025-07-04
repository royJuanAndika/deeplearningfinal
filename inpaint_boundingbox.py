import torch
from diffusers import AutoPipelineForInpainting
from transformers import pipeline as transformers_pipeline # Renamed to avoid confusion
from PIL import Image, ImageDraw, ImageFont
import time
import io

# --- Imports for the Web Server ---
import uvicorn
from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import StreamingResponse

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

# --- NEW: Load Object Detection Model ---
print("Loading Object Detection model (Grounding DINO)...")
# Using the transformers.pipeline to load the detection model
detector_pipeline = transformers_pipeline(
    task="zero-shot-object-detection",
    model="IDEA-Research/grounding-dino-base",
    device="cuda" # Ensure it runs on the GPU
)
ml_models["detection_pipeline"] = detector_pipeline
print("Object Detection model ready.")

print("\nAll models loaded. Server is ready to accept requests.")


# --- 2. FastAPI App and Endpoint Definitions ---
app = FastAPI()

# --- Endpoint for Inpainting ---
@app.post("/inpaint/")
async def run_inpainting(
    prompt: str = Form(...),
    negative_prompt: str = Form(...),
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
        mask_image=mask_image
    ).images[0]

    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    buffer.seek(0)

    end_time = time.time()
    print(f"Inpainting took {end_time - start_time:.2f} seconds.")
    return StreamingResponse(buffer, media_type="image/png")

# --- MODIFIED Endpoint for Text-Based Object Detection ---
@app.post("/detect/")
async def run_detection_and_create_mask(
    text_prompt: str = Form(...),
    confidence_threshold: float = Form(0.5), 
    image_file: UploadFile = File(...)
):
    """
    Receives an image and a text prompt, then returns a black and white MASK image
    where detected objects are white.
    """
    print(f"Received MASK creation request for '{text_prompt}' with threshold: {confidence_threshold}")
    start_time = time.time()

    # Read and prepare the original image to get its dimensions
    image_bytes = await image_file.read()
    init_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    width, height = init_image.size

    # --- THIS IS THE KEY CHANGE ---
    # 1. Create a new, completely black image for the mask.
    #    'L' mode is for 8-bit grayscale pixels (0=black, 255=white).
    mask_image = Image.new("L", (width, height), 0)
    draw = ImageDraw.Draw(mask_image)
    # --- END OF CHANGE ---

    # Get the detection pipeline
    detector = ml_models["detection_pipeline"]
    
    # Run detection on the original image
    detections = detector(init_image, candidate_labels=[text_prompt])

    found_objects = 0
    # Loop through all detections and draw on the MASK image
    for detection in detections:
        score = detection['score']
        
        if score >= confidence_threshold:
            found_objects += 1
            box = detection['box']
            xmin, ymin, xmax, ymax = box['xmin'], box['ymin'], box['xmax'], box['ymax']
            
            # --- THIS IS THE KEY CHANGE ---
            # 2. Draw a FILLED, WHITE rectangle onto the black mask image.
            draw.rectangle([xmin, ymin, xmax, ymax], fill=255)
            # --- END OF CHANGE ---

    # --- THIS IS THE KEY CHANGE ---
    # 3. Save the MASK image to the buffer, not the original image.
    buffer = io.BytesIO()
    mask_image.save(buffer, format="PNG")
    buffer.seek(0)
    # --- END OF CHANGE ---

    end_time = time.time()
    print(f"Created mask with {found_objects} objects. Took {end_time - start_time:.2f} seconds.")
    return StreamingResponse(buffer, media_type="image/png")


# --- 3. Running the Server ---
if __name__ == "__main__":
    # Bind to 127.0.0.1 for use with SSH tunnels or reverse proxies
    uvicorn.run(app, host="127.0.0.2", port=8000)