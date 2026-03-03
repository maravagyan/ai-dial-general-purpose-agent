from typing import Any

from task.tools.deployment.base import DeploymentTool


class ImageGenerationTool(DeploymentTool):
    """
    Calls the DALL-E-3 deployment via DIAL adapter and returns an image attachment.
    """

    def __init__(self, endpoint: str, deployment_name: str = "dall-e-3"):
        super().__init__(endpoint=endpoint, deployment_name=deployment_name)

    @property
    def show_in_stage(self) -> bool:
        return True

    @property
    def name(self) -> str:
        return "image_generation"

    @property
    def description(self) -> str:
        return (
            "Generate an image from a text prompt. Use when the user asks to create an image/picture/illustration. "
            "Optional parameters: size, quality, style."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "prompt": {"type": "string", "description": "Text prompt describing the image to generate."},
                "size": {"type": "string", "description": "Image size, e.g. 1024x1024", "default": "1024x1024"},
                "quality": {"type": "string", "description": "Image quality, e.g. standard or hd", "default": "standard"},
                "style": {"type": "string", "description": "Image style, e.g. vivid or natural", "default": "vivid"},
            },
            "required": ["prompt"],
        }