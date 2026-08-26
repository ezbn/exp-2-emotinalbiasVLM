import base64
import json
import os
import re
import requests

"""
Contient uniquement la mécanique pour discuter avec Ollama. Il ne sait rien du shopping ou de l'expérience. 

Fonctions principales :
De prendre des images et de les encoder en Base64 pour que l'IA puisse les "voir".

De gérer l'historique de la conversation (se souvenir de ce qui a été dit).

D'envoyer les requêtes à l'API locale d'Ollama et de renvoyer la réponse.

De nettoyer et d'extraire proprement le JSON si l'IA bavarde trop."""

def encode_image_to_base64(image_path: str) -> str:
    """Encodes a local image file to Base64 for the Ollama API."""
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode("utf-8")

class VLMClient:
    def __init__(
        self,
        model_name: str = "qwen2.5-vl:7b",
        host: str = "http://localhost:11434",
        temperature: float = 0.2,
    ):
        self.model_name = model_name
        self.host = host
        self.temperature = temperature
        self.history = []

    def reset_chat(self, system_prompt: str):
        """Initializes conversation state with a system instruction."""
        self.history = [{"role": "system", "content": system_prompt}]

    def send_message(self, text: str, image_path: str = None) -> str:
        """Sends a message turn and appends both query and response to chat history."""
        msg = {"role": "user", "content": text}
        if image_path and os.path.exists(image_path):
            msg["images"] = [encode_image_to_base64(image_path)]
        self.history.append(msg)

        payload = {
            "model": self.model_name,
            "messages": self.history,
            "stream": False,
            "options": {"temperature": self.temperature},
        }

        response = requests.post(f"{self.host}/api/chat", json=payload)
        response.raise_for_status()
        content = response.json()["message"]["content"]
        
        # Save assistant response to keep context for the next turn
        self.history.append({"role": "assistant", "content": content})
        return content

def extract_json(response_text: str) -> dict:
    """Robust JSON extraction from LLM response text."""
    try:
        # Pattern 1: JSON markdown code block
        match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", response_text, re.DOTALL)
        if match:
            return json.loads(match.group(1))
        
        # Pattern 2: First outermost JSON brackets
        match_raw = re.search(r"\{[\s\S]*\}", response_text)
        if match_raw:
            return json.loads(match_raw.group(0))
            
        return json.loads(response_text)
    except Exception as e:
        return {"items": [], "error": f"JSON parse error: {str(e)}"}