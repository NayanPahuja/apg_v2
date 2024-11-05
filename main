from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from io import BytesIO
from PIL import Image
import inference

app = FastAPI(title="Underwater Image Enhancement API")


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def root():
    return {"message": "Underwater Image Enhancement API"}


@app.post("/enhance")
async def enhance_image(file: UploadFile = File(...)):
    # Input validation
    if not file.filename.lower().endswith(('.png', '.jpg', '.jpeg')):
        raise HTTPException(400, detail="Invalid file format. Please upload PNG or JPG images.")

    try:
        # Read and process input image
        image_data = await file.read()
        input_image = Image.open(BytesIO(image_data)).convert('RGB')

        # Enhance the image
        enhanced_image = inference.enhance_image(input_image)

        # Save the enhanced image to a BytesIO buffer
        buffer = BytesIO()
        enhanced_image.save(buffer, format='PNG')
        buffer.seek(0)

        return StreamingResponse(buffer, media_type="image/png")

    except Exception as e:
        raise HTTPException(500, detail=f"Error processing image: {str(e)}")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
    