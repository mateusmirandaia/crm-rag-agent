"""Turn CRM records into retrievable documents.

Loaders own the mapping from a source schema to `Document`. Everything downstream
(chunking, embeddings, the agent) only sees `Document`, which is what makes swapping
the JSON sample for a live Dataverse environment a loader-only change.
"""

import json
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, Protocol

DocType = Literal["account", "case", "kb_article"]

PRIORITY_LABELS = {1: "Priority 1", 2: "Priority 2", 3: "Priority 3", 4: "Priority 4"}


@dataclass(frozen=True)
class Document:
    id: str
    type: DocType
    title: str
    text: str
    # One-line summary of structured fields (account, priority, status...). Kept apart
    # from `text` so it can label every chunk without being quoted as content.
    header: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


class Loader(Protocol):
    def load(self) -> Iterable[Document]: ...


def account_to_document(record: dict[str, Any]) -> Document:
    location = f"{record['city']}, {record['country']}"
    products = ", ".join(record.get("products", []))
    return Document(
        id=record["id"],
        type="account",
        title=record["name"],
        text=(
            f"{record['name']} is a {record['tier']} tier {record['industry'].lower()} "
            f"account based in {location}, using {products}.\n\n{record['description']}"
        ),
        header=f"Account, {record['industry']}, {location}.",
        metadata={
            "industry": record["industry"],
            "country": record["country"],
            "tier": record["tier"],
        },
    )


def case_to_document(record: dict[str, Any], account_names: dict[str, str]) -> Document:
    # The account name is copied into the case text on purpose: users ask "what
    # issues did Velmora have?", and a case that only carries ACC-001 would never
    # match that query on embeddings alone.
    account = account_names.get(record["account_id"], record["account_id"])
    priority = PRIORITY_LABELS.get(record["priority"], f"Priority {record['priority']}")
    parts = [f"Problem: {record['description']}"]
    if record.get("resolution"):
        parts.append(f"Resolution: {record['resolution']}")
    return Document(
        id=record["id"],
        type="case",
        title=record["title"],
        text="\n\n".join(parts),
        header=(
            f"Case for {account} about {record['product']}. {priority}, "
            f"status {record['status']}, opened {record['created_on']}."
        ),
        metadata={
            "account_id": record["account_id"],
            "account": account,
            "product": record["product"],
            "status": record["status"],
            "priority": record["priority"],
        },
    )


def article_to_document(record: dict[str, Any]) -> Document:
    return Document(
        id=record["id"],
        type="kb_article",
        title=record["title"],
        text=record["body"],
        header=f"Knowledge article, {record['product']}.",
        metadata={"product": record["product"]},
    )


class JsonDirectoryLoader:
    """Reads the sample export layout: accounts.json, cases.json, knowledge_articles.json."""

    def __init__(self, data_dir: Path) -> None:
        self.data_dir = Path(data_dir)

    def _read(self, name: str) -> list[dict[str, Any]]:
        path = self.data_dir / name
        if not path.exists():
            return []
        with path.open(encoding="utf-8") as fh:
            return json.load(fh)

    def load(self) -> Iterator[Document]:
        accounts = self._read("accounts.json")
        account_names = {a["id"]: a["name"] for a in accounts}
        for record in accounts:
            yield account_to_document(record)
        for record in self._read("cases.json"):
            yield case_to_document(record, account_names)
        for record in self._read("knowledge_articles.json"):
            yield article_to_document(record)


class DataverseLoader:
    """Placeholder for loading straight from Dataverse via the Web API.

    The intended implementation: authenticate with an app registration
    (client credentials via MSAL), page through `accounts`, `incidents` and
    `knowledgearticles` with `$select` limited to the fields the mappers need,
    and follow `@odata.nextLink`. Map rows to the same dict shape the JSON
    mappers accept so `case_to_document` and friends are reused unchanged.
    For incremental ingest, filter on `modifiedon` greater than the last run.
    """

    ENTITY_SETS = ("accounts", "incidents", "knowledgearticles")

    def __init__(
        self, environment_url: str, tenant_id: str, client_id: str, client_secret: str
    ) -> None:
        self.environment_url = environment_url.rstrip("/")
        self.tenant_id = tenant_id
        self.client_id = client_id
        self.client_secret = client_secret

    def load(self) -> Iterator[Document]:
        raise NotImplementedError("DataverseLoader is not implemented yet; see the docstring.")
