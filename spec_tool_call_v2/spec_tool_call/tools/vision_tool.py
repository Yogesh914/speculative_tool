"""Multimodal tool for handling images, audio, and video."""
import os
import base64
from typing import Dict, Any, Optional
from pathlib import Path

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage

from ..prompts import VISION_ANALYZE_PROMPT, VISION_OCR_PROMPT


def analyze_image(image_path: str, question: Optional[str] = None) -> Dict[str, Any]:
    """
    Analyze an image using vision model (GPT-4V).
    
    Args:
        image_path: Path to image file or URL
        question: Optional specific question about the image
        
    Returns:
        Dict with analysis results
    """
    try:
        # Check if file exists (for local files)
        if not image_path.startswith(('http://', 'https://')):
            if not os.path.exists(image_path):
                return {
                    "error": "file_not_found",
                    "path": image_path,
                    "message": f"Image file does not exist: {image_path}"
                }
        
        # Prepare the image content
        if image_path.startswith(('http://', 'https://')):
            # URL image
            image_content = {
                "type": "image_url",
                "image_url": {"url": image_path}
            }
        else:
            # Local file - encode to base64
            image_content = _encode_image_base64(image_path)
        
        # Prepare prompt using template
        if question:
            prompt_text = VISION_ANALYZE_PROMPT.format(query=question)
        else:
            prompt_text = (
                "Please describe this image in detail. Include:\n"
                "1. Main subjects and objects\n"
                "2. Colors, composition, and visual elements\n"
                "3. Any text visible in the image\n"
                "4. Context or setting\n"
                "5. Notable details or features"
            )

        # Create message with image
        message = HumanMessage(
            content=[
                {"type": "text", "text": prompt_text},
                image_content
            ]
        )

        # Call vision model (using GPT-5-mini for efficiency)
        llm = ChatOpenAI(model="gpt-5-mini", max_tokens=2048)
        response = llm.invoke([message])

        return {
            "status": "success",
            "image_path": image_path,
            "question": question,
            "analysis": response.content,
            "model": "gpt-5-mini"
        }
    
    except Exception as e:
        return {
            "status": "error",
            "error": str(e),
            "image_path": image_path
        }


def extract_text_from_image(image_path: str) -> Dict[str, Any]:
    """
    Extract text from an image using OCR via vision model.
    Converts document images to markdown format.

    Args:
        image_path: Path to image file

    Returns:
        Dict with extracted text in markdown format
    """
    try:
        # Check if file exists (for local files)
        if not image_path.startswith(('http://', 'https://')):
            if not os.path.exists(image_path):
                return {
                    "error": "file_not_found",
                    "path": image_path,
                    "message": f"Image file does not exist: {image_path}"
                }

        # Prepare the image content
        if image_path.startswith(('http://', 'https://')):
            # URL image
            image_content = {
                "type": "image_url",
                "image_url": {"url": image_path}
            }
        else:
            # Local file - encode to base64
            image_content = _encode_image_base64(image_path)

        # Use OCR prompt
        prompt_text = VISION_OCR_PROMPT

        # Create message with image
        message = HumanMessage(
            content=[
                {"type": "text", "text": prompt_text},
                image_content
            ]
        )

        # Call vision model (using GPT-5-mini for efficiency)
        llm = ChatOpenAI(model="gpt-5-mini", max_tokens=4096)
        response = llm.invoke([message])

        return {
            "status": "success",
            "image_path": image_path,
            "extracted_text": response.content,
            "method": "vision_ocr",
            "model": "gpt-5-mini"
        }

    except Exception as e:
        return {
            "status": "error",
            "error": str(e),
            "image_path": image_path
        }


def get_image_info(image_path: str) -> Dict[str, Any]:
    """
    Get basic information about an image file.
    
    Args:
        image_path: Path to image file
        
    Returns:
        Dict with image metadata
    """
    try:
        from PIL import Image
        
        if not os.path.exists(image_path):
            return {
                "error": "file_not_found",
                "path": image_path
            }
        
        with Image.open(image_path) as img:
            return {
                "status": "success",
                "path": image_path,
                "format": img.format,
                "mode": img.mode,
                "size": img.size,
                "width": img.width,
                "height": img.height,
                "file_size_bytes": os.path.getsize(image_path)
            }
    
    except ImportError:
        return {
            "error": "missing_dependency",
            "message": "Install Pillow (PIL) for image info extraction"
        }
    except Exception as e:
        return {
            "error": "read_error",
            "message": str(e)
        }


def analyze_audio(audio_path: str) -> Dict[str, Any]:
    """
    Analyze audio file (transcription placeholder).
    
    Args:
        audio_path: Path to audio file
        
    Returns:
        Dict with audio info
    """
    if not os.path.exists(audio_path):
        return {
            "error": "file_not_found",
            "path": audio_path
        }
    
    file_size = os.path.getsize(audio_path)
    file_ext = Path(audio_path).suffix.lower()
    
    return {
        "status": "info",
        "path": audio_path,
        "format": file_ext,
        "size_bytes": file_size,
        "message": "Audio transcription not yet implemented. Use OpenAI Whisper API for transcription.",
        "suggestion": "Consider using OpenAI's Whisper model for audio transcription"
    }


def analyze_video(video_path: str, sample_frames: int = 5) -> Dict[str, Any]:
    """
    Analyze video file by sampling frames.
    
    Args:
        video_path: Path to video file
        sample_frames: Number of frames to sample
        
    Returns:
        Dict with video info
    """
    if not os.path.exists(video_path):
        return {
            "error": "file_not_found",
            "path": video_path
        }
    
    file_size = os.path.getsize(video_path)
    file_ext = Path(video_path).suffix.lower()
    
    return {
        "status": "info",
        "path": video_path,
        "format": file_ext,
        "size_bytes": file_size,
        "message": "Video analysis not yet implemented.",
        "suggestion": "Consider extracting frames and using image analysis, or use specialized video understanding models"
    }


# Helper functions

def _encode_image_base64(image_path: str) -> Dict[str, Any]:
    """Encode local image to base64 for API."""
    with open(image_path, "rb") as image_file:
        image_data = base64.b64encode(image_file.read()).decode('utf-8')
    
    # Detect image type
    ext = Path(image_path).suffix.lower()
    mime_types = {
        '.jpg': 'image/jpeg',
        '.jpeg': 'image/jpeg',
        '.png': 'image/png',
        '.gif': 'image/gif',
        '.webp': 'image/webp'
    }
    mime_type = mime_types.get(ext, 'image/jpeg')
    
    return {
        "type": "image_url",
        "image_url": {
            "url": f"data:{mime_type};base64,{image_data}"
        }
    }


if __name__ == "__main__":
    """Test multimodal capabilities"""
    import sys

    print("=" * 70)
    print("VISION TOOL TEST")
    print("=" * 70)

    # Check if an image path is provided as command line argument
    if len(sys.argv) > 1:
        test_image = sys.argv[1]
    else:
        test_image = os.getenv("TEST_IMAGE_PATH")

    if not test_image or not os.path.exists(test_image):
        print("\n[INFO] No test image provided.")
        print("Usage:")
        print("  python vision_tool.py /path/to/image.png")
        print("  or set TEST_IMAGE_PATH environment variable")
        print("\nSkipping vision tests...")
        sys.exit(0)

    print(f"\nTesting with image: {test_image}")

    # Test 1: Get image info
    print("\n[TEST 1] Get image info:")
    result = get_image_info(test_image)
    print(f"Status: {result.get('status')}")
    if result.get('status') == 'success':
        print(f"  Format: {result.get('format')}")
        print(f"  Size: {result.get('width')}x{result.get('height')}")
        print(f"  Mode: {result.get('mode')}")
        print(f"  File size: {result.get('file_size_bytes')} bytes")
    else:
        print(f"  Error: {result.get('error')}")

    # Test 2: Analyze image (general description)
    print("\n[TEST 2] Analyze image (general description):")
    result = analyze_image(test_image)
    print(f"Status: {result.get('status')}")
    if result.get('status') == 'success':
        print(f"Model: {result.get('model')}")
        analysis = result.get('analysis', '')
        if len(analysis) > 300:
            print(f"Analysis (truncated): {analysis[:300]}...")
        else:
            print(f"Analysis: {analysis}")
    else:
        print(f"Error: {result.get('error')}")

    # Test 3: Analyze image with specific question
    print("\n[TEST 3] Analyze image with specific question:")
    question = "What colors are dominant in this image?"
    result = analyze_image(test_image, question=question)
    print(f"Question: {question}")
    print(f"Status: {result.get('status')}")
    if result.get('status') == 'success':
        print(f"Model: {result.get('model')}")
        analysis = result.get('analysis', '')
        if len(analysis) > 200:
            print(f"Answer (truncated): {analysis[:200]}...")
        else:
            print(f"Answer: {analysis}")
    else:
        print(f"Error: {result.get('error')}")

    # Test 4: Extract text from image (OCR)
    print("\n[TEST 4] Extract text from image (OCR):")
    result = extract_text_from_image(test_image)
    print(f"Status: {result.get('status')}")
    if result.get('status') == 'success':
        print(f"Model: {result.get('model')}")
        print(f"Method: {result.get('method')}")
        extracted = result.get('extracted_text', '')
        if len(extracted) > 300:
            print(f"Extracted text (truncated):\n{extracted[:300]}...")
        else:
            print(f"Extracted text:\n{extracted}")
    else:
        print(f"Error: {result.get('error')}")

    # Test 5: Test with non-existent file
    print("\n[TEST 5] Test error handling (non-existent file):")
    result = analyze_image("/path/to/nonexistent/image.png")
    print(f"Expected error: {result.get('error')}")
    print(f"Message: {result.get('message')}")

    print("\n" + "=" * 70)
    print("ALL TESTS COMPLETED")
    print("=" * 70)
    print("\nNOTE: Vision tests require:")
    print("  - Valid OPENAI_API_KEY environment variable")
    print("  - GPT-5-mini model access")
    print("  - Test image file")

