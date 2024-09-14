from pathlib import Path

import torch
from PIL import Image
from colpali_engine.models import ColPali, ColPaliProcessor


class LitePali:
    def __init__(self, model_name="vidore/colpali-v1.2", device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = ColPali.from_pretrained(
            model_name, torch_dtype=torch.bfloat16, device_map=self.device
        ).eval()
        self.processor = ColPaliProcessor.from_pretrained(model_name)
        self.index = []
        self.file_paths = []

    def add_to_index(self, input_path):
        if Path(input_path).is_dir():
            image_paths = [
                p
                for p in Path(input_path).glob("*")
                if p.suffix.lower()
                in [".jpg", ".jpeg", ".png", ".bmp", ".gif", ".tiff"]
            ]
        else:
            image_paths = [Path(input_path)]

        images = [Image.open(path) for path in image_paths]
        batch_images = self.processor.process_images(images).to(self.model.device)

        with torch.no_grad():
            image_embeddings = self.model(**batch_images)

        self.index.extend(image_embeddings.cpu())
        self.file_paths.extend(map(str, image_paths))
        print(f"Added {len(images)} images to index.")

    def search(self, query, k=5):
        batch_query = self.processor.process_queries([query]).to(self.model.device)
        with torch.no_grad():
            query_embedding = self.model(**batch_query)

        image_embeddings = torch.stack(self.index).to(self.model.device)
        scores = self.processor.score_multi_vector(query_embedding, image_embeddings)[0]

        top_indices = scores.argsort(descending=True)[:k].tolist()
        results = [
            {"path": self.file_paths[i], "score": float(scores[i])} for i in top_indices
        ]
        return results

    def encode_image(self, image):
        batch_image = self.processor.process_images([image]).to(self.model.device)
        with torch.no_grad():
            return self.model(**batch_image).cpu()

    def encode_query(self, query):
        batch_query = self.processor.process_queries([query]).to(self.model.device)
        with torch.no_grad():
            return self.model(**batch_query).cpu()

    def save_index(self, path):
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        torch.save(self.index, path / "index.pt")
        with open(path / "file_paths.txt", "w") as f:
            f.write("\n".join(self.file_paths))

    def load_index(self, path):
        path = Path(path)
        self.index = torch.load(path / "index.pt")
        with open(path / "file_paths.txt", "r") as f:
            self.file_paths = f.read().splitlines()
