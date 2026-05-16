from __future__ import annotations

import os
import logging
from huggingface_hub import HfApi, login

from src.config.settings import HuggingFaceSettings

logger = logging.getLogger(__name__)


class HuggingFacePublisher:

    MODEL_FILENAME = "saved_models/sbert_transformer_clf.pt"
    TOKENIZER_SUBDIR = "saved_models/tokenizer"

    def __init__(self, hub_cfg: HuggingFaceSettings):
        self._hub_cfg = hub_cfg

    def is_enabled(self) -> bool:
        return self._hub_cfg.is_configured

    def push(
        self,
        model_path: str,
        tokenizer_dir: str
    ) -> str:
        if not self.is_enabled():
            raise RuntimeError("HuggingFace publisher not enabled: enter HF_REPO_ID in environment variables.")

        login(token=self._hub_cfg.token, add_to_git_credential=False)
        api = HfApi()

        repo_url = api.create_repo(
            repo_id=self._hub_cfg.repository_id,
            repo_type=self._hub_cfg.repository_type,
            private=not self._hub_cfg.is_public,
            exist_ok=True,
        )
        logger.info(f"HuggingFace: repository {repo_url}")

        if os.path.exists(model_path):
            api.upload_file(
                path_or_fileobj=model_path,
                path_in_repo=self.MODEL_FILENAME,
                repo_id=self._hub_cfg.repository_id,
                repo_type=self._hub_cfg.repository_type,
                commit_message=f"{self._hub_cfg.commit_message} [model weights]"
            )
            logger.info("HuggingFace: model weights uploaded")

        if os.path.isdir(tokenizer_dir):
            api.upload_folder(
                folder_path=tokenizer_dir,
                path_in_repo=self.TOKENIZER_SUBDIR,
                repo_id=self._hub_cfg.repository_id,
                repo_type=self._hub_cfg.repository_type,
                commit_message=f"{self._hub_cfg.commit_message} [tokenizer]"
            )
            logger.info("HuggingFace: tokenizer artifacts uploaded")

        repo_url_str = f"https://huggingface.co/{self._hub_cfg.repository_id}"
        logger.info(f"HuggingFace: uploading completed -> {repo_url_str}")
        return repo_url_str