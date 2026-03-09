import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import requests

PINATA_API_BASE    = "https://api.pinata.cloud"
PINATA_UPLOAD_BASE = "https://uploads.pinata.cloud"
PINATA_JWT_ENV     = "PINATA_JWT"
PINATA_GATEWAY     = "https://gateway.pinata.cloud/ipfs"


@dataclass(frozen=True)
class PinataUploadResult:
    ipfs_cid: str
    size_bytes: int
    timestamp: str
    raw_response: dict[str, Any]


class PinataClient:
    def __init__(
        self,
        jwt_token: str | None = None,
        api_base: str = PINATA_API_BASE,
        upload_base: str = PINATA_UPLOAD_BASE,
        env_var: str = PINATA_JWT_ENV,
    ) -> None:
        token = jwt_token or os.getenv(env_var)
        if not token:
            raise ValueError(
                f"Pinata JWT token is required. Pass jwt_token or set {env_var}."
            )
        self.api_base    = api_base.rstrip("/")
        self.upload_base = upload_base.rstrip("/")
        self._headers    = {"Authorization": f"Bearer {token}"}

    # ── Groups (v3) ────────────────────────────────────────────────────────────

    def create_group(self, name: str) -> str:
        """Create a Pinata group and return its ID."""
        url  = f"{self.api_base}/v3/groups"
        resp = requests.post(
            url,
            headers={**self._headers, "Content-Type": "application/json"},
            json={"name": name},
            timeout=30,
        )
        if resp.status_code >= 400:
            raise RuntimeError(
                f"Pinata create_group failed ({resp.status_code}): {resp.text}"
            )
        return resp.json()["data"]["id"]

    def list_group_files(self, group_id: str, limit: int = 100) -> list[dict[str, Any]]:
        """Return all files belonging to a Pinata group (v3)."""
        url    = f"{self.api_base}/v3/files"
        params = {"group": group_id, "limit": limit}
        resp   = requests.get(url, headers=self._headers, params=params, timeout=30)
        if resp.status_code >= 400:
            raise RuntimeError(
                f"Pinata list_group_files failed ({resp.status_code}): {resp.text}"
            )
        return resp.json()["data"]["files"]

    # ── Upload (v3) ────────────────────────────────────────────────────────────

    def upload_clip(
        self,
        file_path: str | Path,
        *,
        name: str | None = None,
        group_id: str | None = None,
        keyvalues: dict[str, str] | None = None,
        timeout_s: int = 120,
    ) -> PinataUploadResult:
        """Upload a file via the v3 Files API, optionally into a group."""
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"File does not exist: {path}")

        form: dict[str, Any] = {"name": name or path.name}
        if group_id:
            form["group_id"] = group_id
        if keyvalues:
            form["keyvalues"] = json.dumps(keyvalues)

        url = f"{self.upload_base}/v3/files"
        with path.open("rb") as fh:
            response = requests.post(
                url,
                headers=self._headers,
                files={"file": (path.name, fh, "video/mp4")},
                data=form,
                timeout=timeout_s,
            )

        if response.status_code >= 400:
            raise RuntimeError(
                f"Pinata upload failed ({response.status_code}): {response.text}"
            )

        data = response.json()["data"]
        return PinataUploadResult(
            ipfs_cid=data["cid"],
            size_bytes=int(data.get("size", 0)),
            timestamp=str(data.get("created_at", "")),
            raw_response=data,
        )
