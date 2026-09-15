from datasets import load_dataset


DATASET_METADATA = {
    "fineweb": {
        "path": "HuggingFaceFW/fineweb-edu",
        "name": "sample-10BT",
        "default_revision": "main",
    },
    "tinystories": {
        "path": "roneneldan/TinyStories",
        "name": None,
        "default_revision": "main",
    },
    "local": {
        "path": "local text files",
        "name": None,
        "default_revision": "user-provided",
    },
}


def dataset_metadata(dataset_name, revision=None, mixture=None, local_files=None):
    meta = dict(DATASET_METADATA[dataset_name])
    meta["revision"] = revision or meta["default_revision"]
    meta["mixture"] = mixture or f"{dataset_name}:1.0"
    meta["local_files"] = local_files or []
    return meta


def fineweb_stream(split="train", revision=None):
    dataset = load_dataset(
        "HuggingFaceFW/fineweb-edu",
        name="sample-10BT",
        split=split,
        streaming=True,
        revision=revision,
    )
    for row in dataset:
        text = row.get("text", "")
        if text:
            yield text


def tinystories_stream(split="train", revision=None):
    dataset = load_dataset("roneneldan/TinyStories", split=split, streaming=True, revision=revision)
    for row in dataset:
        text = row.get("text", "")
        if text:
            yield text


def local_text_stream(paths):
    for path in paths:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    yield line


def choose_text_stream(dataset_name, local_files=None, revision=None):
    if dataset_name == "tinystories":
        return tinystories_stream(revision=revision)
    if dataset_name == "fineweb":
        return fineweb_stream(revision=revision)
    if dataset_name == "local":
        return local_text_stream(local_files or [])
    raise ValueError(f"Unknown dataset_name: {dataset_name}")


def token_batches(text_iter, tokenizer, seq_len, batch_size):
    buffer = []
    eos = tokenizer.eos_token_id
    for text in text_iter:
        ids = tokenizer.encode(text, add_special_tokens=False) + [eos]
        buffer.extend(ids)
        while len(buffer) >= batch_size * seq_len:
            block = buffer[: batch_size * seq_len]
            buffer = buffer[batch_size * seq_len :]
            yield [block[i : i + seq_len] for i in range(0, len(block), seq_len)]
