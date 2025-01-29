#!/bin/bash

# Directory containing PDF files
INPUT_DIR="/home/yp6443/research/nlp/voice_data/val_file/pdfs"
# Output directory for text files
OUTPUT_DIR="/home/yp6443/research/nlp/voice_data/val_file/txts"

# Create output directory if it doesn't exist
mkdir -p "$OUTPUT_DIR"

# Loop through all PDF files in the input directory
for PDF_FILE in "$INPUT_DIR"/*.pdf; do
    # Get the base name of the PDF file (without extension)
    BASENAME=$(basename "$PDF_FILE" .pdf)
    
    # Convert PDF to images (one image per page)
    pdftoppm -png "$PDF_FILE" "$OUTPUT_DIR/$BASENAME"
    
    # Run Tesseract on each image and save the text
    for IMAGE in "$OUTPUT_DIR/$BASENAME"*.png; do
        tesseract "$IMAGE" "${IMAGE%.*}"
    done
    
    # Combine all text files for this PDF into one
    cat "$OUTPUT_DIR/$BASENAME"*.txt > "$OUTPUT_DIR/$BASENAME.txt"
    
    # Optionally, remove intermediate image and text files
    rm "$OUTPUT_DIR/$BASENAME"*.png
    rm "$OUTPUT_DIR/$BASENAME"-[0-9]*.txt

done

echo "All PDFs processed. Text files saved in $OUTPUT_DIR."