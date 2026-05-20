import os
import pdfplumber

output_dir = "extracted_images"
os.makedirs(output_dir, exist_ok=True)

with pdfplumber.open("data/catalog.pdf") as pdf:
    for page_num, page in enumerate(pdf.pages):
        page_images = page.images

        print(f"\n--- Processing Page {page_num + 1} ---")
        print(f"Images found on this page: {len(page_images)}")

        if not page_images:
            print("No images found on this page.")
            continue

        # Select the largest image by area
        largest_img = max(
            page_images,
            key=lambda img: img.get("width", 0) * img.get("height", 0)
        )

        bbox = (
            largest_img["x0"],
            largest_img["top"],
            largest_img["x1"],
            largest_img["bottom"]
        )

        out_path = os.path.join(
            output_dir,
            f"page_{page_num + 1}_largest_image.png"
        )

        cropped_page = page.crop(bbox)
        cropped_page.to_image(resolution=150).save(out_path, format="JPEG")

        area = largest_img.get("width", 0) * largest_img.get("height", 0)
        print(f"Selected largest image on Page {page_num + 1}")
        print(f"Width: {largest_img.get('width')}, Height: {largest_img.get('height')}, Area: {area}")
        print(f"Saved to: {out_path}")