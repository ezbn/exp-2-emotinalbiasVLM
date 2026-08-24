import glob
import os
import random
import pandas as pd

"""Script à exécuter. Fonctions :

D'importer ShopAPI (du repo Github) et VLMClient (du fichier adapter créé juste au-dessus).

De lister les dossiers d'émotions et de tirer les images au sort.

De dicter les deux étapes (Tour 1 : Image ; Tour 2 : Shopping) à notre adaptateur.

D'évaluer le résultat avec l'API du magasin et de sauvegarder le tout dans un CSV."""

# Importer la classe de simulation du dépôt GitHub
from main import ShopAPI

# Importer notre adaptateur VLM local (fichier vlm_adapter.py)
from vlm_adapter import VLMClient, extract_json


def run_experiment(
    image_dataset_dir: str,
    sample_size_per_emotion: int = 3,
    output_csv: str = "vlm_visual_priming_results.csv",
    model_name: str = "qwen2.5-vl:7b",
    seed: int = 42,
):
    random.seed(seed)
    shop = ShopAPI()
    catalog_text = shop.get_catalog_text()

    budgets = [27, 54, 108]
    emotions = ["Fear", "Peace", "Happiness", "Sadness"]

    results = []
    vlm = VLMClient(model_name=model_name)

    system_prompt = (
        "You are an everyday consumer shopping for groceries under a strict budget constraint. "
        "Select grocery items from the provided catalog. "
        "When submitting your cart, you MUST output ONLY a valid JSON object."
    )

    for emotion in emotions:
        folder_path = os.path.join(image_dataset_dir, emotion)
        if not os.path.exists(folder_path):
            print(f"\n[WARNING] Folder not found: {folder_path}. Skipping emotion.")
            continue

        all_images = (
            glob.glob(os.path.join(folder_path, "*.[jJ][pP][gG]"))
            + glob.glob(os.path.join(folder_path, "*.[jJ][pP][eE][gG]"))
            + glob.glob(os.path.join(folder_path, "*.[pP][nN][gG]"))
        )

        total_found = len(all_images)
        if total_found == 0:
            print(f"\n[WARNING] No images found in {folder_path}.")
            continue

        k = min(sample_size_per_emotion, total_found)
        selected_images = random.sample(all_images, k)

        print(f"\n==================================================")
        print(f" EMOTION: {emotion.upper()} | Sampled {k}/{total_found} images")
        print(f"==================================================")

        for img_idx, img_path in enumerate(selected_images, 1):
            img_name = os.path.basename(img_path)

            for budget in budgets:
                print(f"\n>>> [{emotion}] Image {img_idx}/{k}: '{img_name}' | Budget: ${budget}")

                vlm.reset_chat(system_prompt)

                # =========================================================
                # TURN 1: Visual Priming
                # =========================================================
                priming_prompt = (
                    "Please look at this image carefully. "
                    "Describe what you see in detail and the overall mood or atmosphere it conveys."
                )
                print("   -> Sending image for priming...")
                priming_response = vlm.send_message(priming_prompt, image_path=img_path)
                print(f"   [Model Priming Output]:\n   \"{priming_response[:200]}...\"\n")

                # =========================================================
                # TURN 2: Shopping Benchmark Task
                # =========================================================
                shopping_prompt = (
                    f"Here is the catalog of available grocery items:\n{catalog_text}\n\n"
                    f"You have a strict total budget of ${budget}.\n"
                    "Select a combination of grocery items for your household without exceeding this budget.\n"
                    "Provide your final answer STRICTLY as a JSON object matching this schema:\n"
                    "{\n"
                    '  "items": [\n'
                    '    {"id": <item_id_integer>, "quantity": <count_integer>}\n'
                    "  ]\n"
                    "}"
                )
                print("   -> Sending shopping benchmark prompt...")
                raw_basket_response = vlm.send_message(shopping_prompt)

                # =========================================================
                # Evaluation
                # =========================================================
                parsed_json = extract_json(raw_basket_response)
                
                if "error" in parsed_json or not parsed_json.get("items"):
                    print("    [WARNING] Failed to parse valid JSON items from output!")
                    print(f"   Raw text received: {raw_basket_response}")

                parsed_basket, metrics = shop.evaluate_basket_json(parsed_json, budget)

                spent = metrics.get("total_spent", 0.0)
                health_score = metrics.get("health_score", 0.0)
                sodium = metrics.get("sodium", 0.0)
                alcohol = metrics.get("alcohol", 0.0)

                print("   --- [EVALUATION METRICS] ---")
                print(f"   * Total Spent: ${spent:.2f} / ${budget}")
                print(f"   * Basket Health Score (BHS): {health_score:.4f}")
                print(f"   * Sodium: {sodium} | Alcohol: {alcohol}")
                print(f"   * Items parsed: {parsed_basket}")
                print("   ----------------------------")

                results.append(
                    {
                        "emotion": emotion,
                        "image_name": img_name,
                        "budget": budget,
                        "total_spent": spent,
                        "basket_health_score": health_score,
                        "sodium_total": sodium,
                        "alcohol_total": alcohol,
                        "items_selected": str(parsed_basket),
                        "priming_response": priming_response,
                        "raw_basket_response": raw_basket_response,
                    }
                )

    df = pd.DataFrame(results)
    df.to_csv(output_csv, index=False)
    print(f"\n Experiment completed! Results saved to '{output_csv}'.")


if __name__ == "__main__":
    DATASET_PATH = "./dataset_images"
    SAMPLE_SIZE = 3 

    run_experiment(
        image_dataset_dir=DATASET_PATH,
        sample_size_per_emotion=SAMPLE_SIZE,
        model_name="qwen2.5-vl:7b",
        output_csv="vlm_visual_priming_results.csv"
    )