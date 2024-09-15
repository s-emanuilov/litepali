import io
import json
import zipfile
from pathlib import Path
from typing import List, Dict, Optional, Union, Tuple

import torch
from PIL import Image
from colpali_engine.models import ColPali, ColPaliProcessor


class ImageFileError(Exception):
    """Custom exception for ImageFile related errors."""

    pass


class LitePaliError(Exception):
    """Custom exception for LitePali related errors."""

    pass


class ImageFile:
    def __init__(
        self,
        path: str,
        document_id: Optional[str] = None,
        page_id: Optional[str] = None,
        metadata: Optional[Dict] = None,
    ):
        if not Path(path).exists():
            raise ImageFileError(f"Image file not found: {path}")
        self.path = path
        self.document_id = document_id
        self.page_id = page_id
        self.metadata = metadata or {}


class LitePali:
    def __init__(self, model_name: str = "vidore/colpali-v1.2", device: Optional[str] = None):
        try:
            self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
            self.model = ColPali.from_pretrained(
                model_name, torch_dtype=torch.bfloat16, device_map=self.device
            ).eval()
            self.processor = ColPaliProcessor.from_pretrained(model_name)
            self.image_embeddings: List[Tuple[ImageFile, Optional[torch.Tensor]]] = []
        except Exception as e:
            raise LitePaliError(f"Failed to initialize LitePali: {str(e)}")

    def add(self, image_file: ImageFile) -> None:
        try:
            self.image_embeddings.append((image_file, None))
        except Exception as e:
            raise LitePaliError(f"Failed to add image file: {str(e)}")

    def process(self, batch_size: int = 32) -> None:
        unprocessed = [item for item in self.image_embeddings if item[1] is None]
        if not unprocessed:
            print("No new images to process.")
            return

        total_processed = 0
        for i in range(0, len(unprocessed), batch_size):
            batch = unprocessed[i : i + batch_size]
            try:
                images = [Image.open(item[0].path) for item in batch]
                batch_images = self.processor.process_images(images).to(self.model.device)

                with torch.no_grad():
                    embeddings = self.model(**batch_images)

                for j, (image_file, _) in enumerate(batch):
                    idx = self.image_embeddings.index((image_file, None))
                    self.image_embeddings[idx] = (image_file, embeddings[j].cpu())

                total_processed += len(batch)
                print(f"Processed batch {i // batch_size + 1}: {total_processed}/{len(unprocessed)} images")
            except Exception as e:
                print(f"Error processing batch: {str(e)}")

        print(f"Finished processing. Total images processed: {total_processed}")

    def add_process(self, image_file: ImageFile, batch_size: int = 32) -> None:
        self.add(image_file)
        self.process(batch_size)

    def search(self, query: str, k: int = 5) -> List[Dict[str, Union[ImageFile, float]]]:
        try:
            batch_query = self.processor.process_queries([query]).to(self.model.device)
            with torch.no_grad():
                query_embedding = self.model(**batch_query)

            processed_embeddings = [
                (i, emb) for i, (_, emb) in enumerate(self.image_embeddings) if emb is not None
            ]
            if not processed_embeddings:
                return []

            indices, embeddings = zip(*processed_embeddings)
            image_embeddings = torch.stack(embeddings).to(self.model.device)
            scores = self.processor.score_multi_vector(query_embedding, image_embeddings)[0]

            top_k = min(k, len(scores))
            top_indices = scores.argsort(descending=True)[:top_k].tolist()
            results = [
                {"image": self.image_embeddings[indices[i]][0], "score": float(scores[i])}
                for i in top_indices
            ]
            return results
        except Exception as e:
            raise LitePaliError(f"Search failed: {str(e)}")

    def save_index(self, path: str, compressed: bool = False) -> None:
        try:
            path = Path(path)
            path.mkdir(parents=True, exist_ok=True)

            embeddings = []
            metadata = []

            for i, (img, emb) in enumerate(self.image_embeddings):
                metadata.append(
                    {
                        "index": i,
                        "path": img.path,
                        "document_id": img.document_id,
                        "page_id": img.page_id,
                        "metadata": img.metadata,
                        "has_embedding": emb is not None,
                    }
                )
                if emb is not None:
                    embeddings.append(emb)

            if compressed:
                with zipfile.ZipFile(path / "index.zip", "w", zipfile.ZIP_DEFLATED) as zipf:
                    if embeddings:
                        buffer = io.BytesIO()
                        torch.save(torch.stack(embeddings), buffer)
                        zipf.writestr("embeddings.pt", buffer.getvalue())
                    zipf.writestr("metadata.json", json.dumps(metadata))
                print(f"Compressed index saved to {path / 'index.zip'}")
            else:
                if embeddings:
                    torch.save(torch.stack(embeddings), path / "embeddings.pt")
                with open(path / "metadata.json", "w") as f:
                    json.dump(metadata, f)
                print(f"Index saved to {path}")
        except Exception as e:
            raise LitePaliError(f"Failed to save index: {str(e)}")

    def load_index(self, path: str) -> None:
        try:
            path = Path(path)

            if path.is_file() and path.suffix == ".zip":
                with zipfile.ZipFile(path, "r") as zipf:
                    with zipf.open("metadata.json") as f:
                        metadata = json.load(f)
                    if "embeddings.pt" in zipf.namelist():
                        with zipf.open("embeddings.pt") as f:
                            buffer = io.BytesIO(f.read())
                            embeddings = torch.load(buffer)
                    else:
                        embeddings = []
            else:
                with open(path / "metadata.json", "r") as f:
                    metadata = json.load(f)
                if (path / "embeddings.pt").exists():
                    embeddings = torch.load(path / "embeddings.pt")
                else:
                    embeddings = []

            self.image_embeddings = []
            embedding_index = 0

            for item in sorted(metadata, key=lambda x: x["index"]):
                image_file = ImageFile(
                    path=item["path"],
                    document_id=item["document_id"],
                    page_id=item["page_id"],
                    metadata=item["metadata"],
                )
                if item["has_embedding"]:
                    embedding = embeddings[embedding_index]
                    embedding_index += 1
                else:
                    embedding = None
                self.image_embeddings.append((image_file, embedding))

            print(f"Index loaded from {path}")
        except Exception as e:
            raise LitePaliError(f"Failed to load index: {str(e)}")

    def index_stats(self) -> Dict[str, Union[int, List[str]]]:
        try:
            return {
                "total_images": len(self.image_embeddings),
                "processed_images": sum(1 for _, emb in self.image_embeddings if emb is not None),
                "unique_documents": len(
                    set(img.document_id for img, _ in self.image_embeddings if img.document_id)
                ),
                "image_extensions": list(set(Path(img.path).suffix for img, _ in self.image_embeddings)),
            }
        except Exception as e:
            raise LitePaliError(f"Failed to generate index stats: {str(e)}")
