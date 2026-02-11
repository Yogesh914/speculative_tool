VISION_ANALYZE_PROMPT = """
You are analyzing an image to answer a specific question.

Question: {query}

Please analyze the image carefully and provide a detailed answer to the question above.
Focus on the relevant visual elements, text, objects, or patterns that help answer the question.
Be precise and thorough in your response.
"""


VISION_OCR_PROMPT = """
Convert this document to pure text markdown.

Extract ALL text visible in the image and format it as markdown.
Preserve the structure, headings, lists, tables, and formatting as much as possible.
If there are multiple columns, transcribe left to right, top to bottom.
"""