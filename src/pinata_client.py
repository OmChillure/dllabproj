import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import requests


PINATA_API_BASE = "https://api.pinata.cloud"
PINATA_JWT_ENV = "PINATA_JWT"


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
        env_var: str = PINATA_JWT_ENV,
    ) -> None:
        token = jwt_token or os.getenv(env_var)
        if not token:
            raise ValueError(
                f"Pinata JWT token is required. Pass jwt_token or set {env_var}."
            )
        self.api_base = api_base.rstrip("/")
        self._headers = {"Authorization": f"Bearer {token}"}

    def upload_file(
        self,
        file_path: str | Path,
        *,
        name: str | None = None,
        keyvalues: dict[str, str] | None = None,
        cid_version: int = 1,
        timeout_s: int = 120,
    ) -> PinataUploadResult:
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"File does not exist: {path}")

        metadata = {"name": name or path.name}
        if keyvalues:
            metadata["keyvalues"] = keyvalues

        options = {"cidVersion": cid_version}
        url = f"{self.api_base}/pinning/pinFileToIPFS"

        with path.open("rb") as file_handle:
            files = {"file": (path.name, file_handle)}
            data = {
                "pinataMetadata": json.dumps(metadata),
                "pinataOptions": json.dumps(options),
            }
            response = requests.post(
                url,
                headers=self._headers,
                files=files,
                data=data,
                timeout=timeout_s,
            )

        if response.status_code >= 400:
            raise RuntimeError(
                f"Pinata file upload failed ({response.status_code}): {response.text}"
            )

        payload = response.json()
        return PinataUploadResult(
            ipfs_cid=payload["IpfsHash"],
            size_bytes=int(payload.get("PinSize", 0)),
            timestamp=str(payload.get("Timestamp", "")),
            raw_response=payload,
        )

    def upload_json(
        self,
        content: dict[str, Any],
        *,
        name: str = "incident-metadata",
        keyvalues: dict[str, str] | None = None,
        cid_version: int = 1,
        timeout_s: int = 60,
    ) -> PinataUploadResult:
        url = f"{self.api_base}/pinning/pinJSONToIPFS"
        body: dict[str, Any] = {
            "pinataOptions": {"cidVersion": cid_version},
            "pinataMetadata": {"name": name},
            "pinataContent": content,
        }
        if keyvalues:
            body["pinataMetadata"]["keyvalues"] = keyvalues

        response = requests.post(
            url,
            headers={**self._headers, "Content-Type": "application/json"},
            data=json.dumps(body),
            timeout=timeout_s,
        )

        if response.status_code >= 400:
            raise RuntimeError(
                f"Pinata JSON upload failed ({response.status_code}): {response.text}"
            )

        payload = response.json()
        return PinataUploadResult(
            ipfs_cid=payload["IpfsHash"],
            size_bytes=int(payload.get("PinSize", 0)),
            timestamp=str(payload.get("Timestamp", "")),
            raw_response=payload,
        )
